"""Decode-node admission control for the Exploring HBF paper comparison.

This module has no media or GPU timing formula.  Its callbacks must schedule
real transfers, writes, kernels and shared-resource competition.  In particular,
initial KV and replacement KV are never installed merely by assigning a length.
Windows contain complete decode events; startup PD imports are reported
separately.  No token/event is skipped and an observed window is not a proof of
periodic steady state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Callable, Mapping

NS_PER_SECOND = 1_000_000_000
THREE_YEARS_SECONDS = 3 * 365 * 24 * 60 * 60


def _integer(name: str, value: int, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")


def _costs(values: Mapping[str, float]) -> None:
    if not isinstance(values, Mapping):
        raise ValueError("costs must map resource names to service nanoseconds")
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("cost names must be non-empty strings")
        _number("cost " + key, value)


@dataclass(frozen=True)
class ServingRequest:
    request_id: int
    input_tokens: int
    output_tokens: int
    generated_decode_tokens: int = 0
    prefill_first_output_token: bool = True
    slot_id: int = 0

    def __post_init__(self) -> None:
        _integer("request_id", self.request_id)
        _integer("input_tokens", self.input_tokens, minimum=1)
        _integer("output_tokens", self.output_tokens, minimum=1)
        _integer("generated_decode_tokens", self.generated_decode_tokens)
        _integer("slot_id", self.slot_id)
        if not isinstance(self.prefill_first_output_token, bool):
            raise ValueError("prefill_first_output_token must be boolean")
        if self.remaining_decode_tokens < 1:
            raise ValueError("request must still have a decode token to generate")

    @property
    def context_tokens(self) -> int:
        """KV tokens resident before the next decode iteration."""
        return self.input_tokens + self.generated_decode_tokens

    @property
    def remaining_decode_tokens(self) -> int:
        return self.output_tokens - int(self.prefill_first_output_token) - self.generated_decode_tokens


@dataclass(frozen=True)
class PendingAdmission:
    request: ServingRequest
    ready_ns: float
    logical_kv_bytes: int
    physical_program_bytes: int
    transfer_bytes: int
    costs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ServingRequest):
            raise ValueError("request must be ServingRequest")
        _number("ready_ns", self.ready_ns)
        for name in ("logical_kv_bytes", "physical_program_bytes", "transfer_bytes"):
            _integer(name, getattr(self, name))
        if not self.logical_kv_bytes or not self.transfer_bytes:
            raise ValueError("PD import must account non-zero KV and transfer bytes")
        _costs(self.costs)


@dataclass(frozen=True)
class DecodeExecution:
    end_ns: float
    physical_program_bytes: int = 0
    logical_write_bytes: int = 0
    transfer_bytes: int = 0
    costs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _number("end_ns", self.end_ns)
        for name in ("physical_program_bytes", "logical_write_bytes", "transfer_bytes"):
            _integer(name, getattr(self, name))
        _costs(self.costs)


@dataclass(frozen=True)
class ServingConfig:
    batch_size: int
    input_tokens: int
    output_tokens: int
    warmup_steps: int = 32
    measurement_steps: int = 128
    prefetch_slots: int = 1
    import_lookahead_tokens: int = 1
    arrival_policy: str = "completion_matched"
    arrival_times_ns: tuple[float, ...] = ()
    full_batch_policy: str = "wait"
    stagger_initial_ages: bool = True
    prefill_first_output_token: bool = True
    start_ns: float = 0.0
    hbf_capacity_bytes: int = 0
    retain_step_records: bool = True

    def __post_init__(self) -> None:
        for name in ("batch_size", "input_tokens", "output_tokens", "measurement_steps"):
            _integer(name, getattr(self, name), minimum=1)
        for name in ("warmup_steps", "prefetch_slots", "import_lookahead_tokens", "hbf_capacity_bytes"):
            _integer(name, getattr(self, name))
        _number("start_ns", self.start_ns)
        if self.output_tokens <= int(self.prefill_first_output_token):
            raise ValueError("output length must include at least one decode token")
        if self.prefetch_slots > self.batch_size:
            raise ValueError("prefetch_slots cannot exceed batch_size")
        if self.arrival_policy not in {"completion_matched", "exogenous"}:
            raise ValueError("unknown arrival_policy")
        if self.full_batch_policy not in {"wait", "partial"}:
            raise ValueError("full_batch_policy must be wait or partial")
        if not all(isinstance(getattr(self, name), bool) for name in (
                "stagger_initial_ages", "prefill_first_output_token", "retain_step_records")):
            raise ValueError("boolean serving options must be boolean")
        previous = self.start_ns
        for arrival in self.arrival_times_ns:
            _number("arrival time", arrival)
            if arrival < previous:
                raise ValueError("arrival times must be ordered and not before start_ns")
            previous = arrival
        if self.arrival_policy == "completion_matched" and self.arrival_times_ns:
            raise ValueError("exogenous arrivals cannot be supplied to completion-matched policy")


ImportCallback = Callable[[ServingRequest, float], PendingAdmission]
DecodeCallback = Callable[[tuple[ServingRequest, ...], tuple[PendingAdmission, ...], float], DecodeExecution]


@dataclass
class ServingResult:
    status: str
    config: dict
    startup_end_ns: float
    startup_import_bytes: int
    startup_program_bytes: int
    startup_costs_ns: dict[str, float]
    measurement_start_ns: float
    measurement_end_ns: float
    warmup_steps_completed: int
    measured_steps: int
    decode_tokens: int
    completed_requests: int
    completed_request_elapsed_count: int
    mean_completed_request_elapsed_ns: float | None
    max_completed_request_elapsed_ns: float | None
    admitted_requests: int
    decode_tokens_per_second: float
    output_equivalent_tokens_per_second: float
    mean_tpot_ns: float
    max_tpot_ns: float
    mean_iteration_ns: float
    max_iteration_ns: float
    mean_batch_size: float
    full_batch_step_fraction: float
    admission_stall_ns: float
    pd_import_bytes: int
    pd_program_bytes: int
    pd_transfer_issued_bytes: int
    pd_program_issued_bytes: int
    pd_transfer_completed_bytes: int
    pd_program_completed_bytes: int
    pending_unfinished_transfer_start_bytes: int
    pending_unfinished_program_start_bytes: int
    pending_unfinished_transfer_end_bytes: int
    pending_unfinished_program_end_bytes: int
    decode_program_bytes: int
    logical_decode_write_bytes: int
    decode_transfer_bytes: int
    pd_resource_service_ns: dict[str, float]
    resource_service_ns: dict[str, float]
    projected_three_year_physical_bytes: float
    projected_three_year_average_pec: float | None
    pending_at_measurement_start: int
    pending_at_measurement_end: int
    step_records: list[dict]
    assumptions: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def run_serving(config: ServingConfig, execute_decode: DecodeCallback,
                import_kv: ImportCallback) -> ServingResult:
    """Execute one independent PD decode node through exact event callbacks.

    ``import_kv`` reserves the actual transfer/write resources immediately and
    returns their completion time; it must include capacity for prepared KV.
    ``execute_decode`` uses the same resource calendars, so imports can overlap
    or delay reads/compute according to the selected hardware.  Cost maps are
    service occupancies, not additive wall-clock latency.  Import traffic is
    reported at admission, issue and whole-import completion independently.
    PEC uses the last quantity (startup is separate).  The callback's import
    completion is the accounting event for its complete program-byte group;
    sub-transfer/program-completion distributions remain the callback's domain.
    """
    if not isinstance(config, ServingConfig):
        raise TypeError("config must be ServingConfig")
    now = float(config.start_ns)
    request_id = 0
    active: list[ServingRequest] = []
    pending: list[PendingAdmission] = []
    last_output: dict[int, float] = {}
    # Only fresh imports admitted by the running controller have a complete
    # measured residence interval. Synthetic staggered startup ages do not.
    fresh_admitted_at: dict[int, float] = {}
    arrival_index = 0
    arrival_backlog = 0
    startup_bytes = startup_program = 0
    startup_costs: dict[str, float] = {}
    measuring_imports = False
    pd_transfer_issued = pd_program_issued = 0
    pd_costs_issued: dict[str, float] = {}

    def add_costs(target: dict[str, float], source: Mapping[str, float]) -> None:
        for key, value in source.items():
            target[key] = target.get(key, 0.) + value

    def schedule_import(at: float, age: int = 0, slot_id: int = 0) -> PendingAdmission:
        nonlocal request_id, pd_transfer_issued, pd_program_issued
        request = ServingRequest(request_id, config.input_tokens, config.output_tokens,
                                 age, config.prefill_first_output_token, slot_id)
        request_id += 1
        imported = import_kv(request, at)
        if not isinstance(imported, PendingAdmission) or imported.request != request:
            raise ValueError("import callback must return the same request and accounted transfer")
        if imported.ready_ns <= at:
            raise ValueError("PD transfer/write must have positive elapsed completion cost")
        if measuring_imports:
            pd_transfer_issued += imported.transfer_bytes
            pd_program_issued += imported.physical_program_bytes
            add_costs(pd_costs_issued, imported.costs)
        return imported

    if config.arrival_policy == "completion_matched":
        length = config.output_tokens - int(config.prefill_first_output_token)
        for index in range(config.batch_size):
            age = index * length // config.batch_size if config.stagger_initial_ages else 0
            imported = schedule_import(now, age, index)
            pending.append(imported)
            startup_bytes += imported.transfer_bytes
            startup_program += imported.physical_program_bytes
            add_costs(startup_costs, imported.costs)
        now = max(row.ready_ns for row in pending)
        active = [row.request for row in pending]
        pending.clear()
        last_output = {row.request_id: now for row in active}
    startup_end = now
    measurement_start = now
    measurement_end = now
    pending_start = 0
    unfinished_transfer_start = unfinished_program_start = 0
    unfinished_costs_start: dict[str, float] = {}
    tokens = completed = admitted = measured = warmup_completed = full = 0
    completed_elapsed_count = 0
    completed_elapsed_sum = 0.
    completed_elapsed_max: float | None = None
    total_intervals = max_interval = max_iteration = batch_sum = stall = 0.
    pd_bytes = pd_program = decode_program = logical_write = transfer = 0
    resource_costs: dict[str, float] = {}
    records: list[dict] = []
    status = "completed"

    for step_index in range(config.warmup_steps + config.measurement_steps):
        measuring = step_index >= config.warmup_steps
        if step_index == config.warmup_steps:
            measurement_start = now
            pending_start = len(pending)
            unfinished_transfer_start = sum(row.transfer_bytes for row in pending if row.ready_ns > now)
            unfinished_program_start = sum(row.physical_program_bytes for row in pending if row.ready_ns > now)
            for row in pending:
                if row.ready_ns > now:
                    add_costs(unfinished_costs_start, row.costs)
            measuring_imports = True
            # First measured interval starts at the window boundary; intervals
            # spanning warmup must not contaminate a finite measured window.
            last_output = {row.request_id: now for row in active}
        iteration_start = now
        iteration_admissions = []

        while True:
            if config.arrival_policy == "exogenous":
                while (arrival_index < len(config.arrival_times_ns)
                       and config.arrival_times_ns[arrival_index] <= now):
                    # Admission scheduling cannot time-travel inside a decode
                    # event. Arrivals during it queue until this boundary.
                    arrival_backlog += 1
                    arrival_index += 1

            ready = [row for row in pending if row.ready_ns <= now]
            ready.sort(key=lambda row: (row.ready_ns, row.request.request_id))
            occupied_slots = {row.slot_id for row in active}
            for imported in ready:
                # A ready successor owns its preselected slot. It cannot fill
                # another vacancy and silently move its KV between EP ranks.
                if imported.request.slot_id not in occupied_slots:
                    pending.remove(imported)
                    active.append(imported.request)
                    occupied_slots.add(imported.request.slot_id)
                    last_output[imported.request.request_id] = now
                    fresh_admitted_at[imported.request.request_id] = now
                    iteration_admissions.append(imported)

            reserved_slots = {row.request.slot_id for row in pending}
            vacant_slots = [slot for slot in range(config.batch_size)
                            if slot not in occupied_slots and slot not in reserved_slots]
            preparing_occupied = len(occupied_slots & reserved_slots)
            available_prefetch = max(0, config.prefetch_slots - preparing_occupied)
            soon = sorted((row for row in active if row.slot_id not in reserved_slots
                           and (config.arrival_policy == "exogenous"
                                or row.remaining_decode_tokens <= config.import_lookahead_tokens)),
                          key=lambda row: (row.remaining_decode_tokens, row.slot_id))
            target_slots = vacant_slots + [row.slot_id for row in soon[:available_prefetch]]
            if config.arrival_policy == "exogenous":
                # Waiting external requests own no KV yet. Each preparation
                # binds one to a known free/soon-to-be-replaced slot.
                target_slots = target_slots[:arrival_backlog]
            for slot in target_slots:
                pending.append(schedule_import(now, slot_id=slot))
                if config.arrival_policy == "exogenous":
                    arrival_backlog -= 1

            if len(active) == config.batch_size:
                break
            if active and config.full_batch_policy == "partial":
                break
            future = [row.ready_ns for row in pending if row.ready_ns > now]
            if config.arrival_policy == "exogenous" and arrival_index < len(config.arrival_times_ns):
                future.append(config.arrival_times_ns[arrival_index])
            if not future:
                break
            now = min(future)

        if not active:
            status = "arrivals_exhausted"
            if measuring:
                measurement_end = now
            break
        decode_start = now
        execution = execute_decode(tuple(active), tuple(pending), now)
        if not isinstance(execution, DecodeExecution) or execution.end_ns <= now:
            raise ValueError("decode callback must return a finite, strictly later completion")
        now = execution.end_ns
        step_tokens = len(active)
        step_completed = 0
        survivors = []
        intervals = []
        for request in active:
            intervals.append(now - last_output[request.request_id])
            last_output[request.request_id] = now
            if request.remaining_decode_tokens == 1:
                step_completed += 1
                del last_output[request.request_id]
                admission_time = fresh_admitted_at.pop(request.request_id, None)
                if measuring and admission_time is not None:
                    elapsed_request = now - admission_time
                    completed_elapsed_count += 1
                    completed_elapsed_sum += elapsed_request
                    completed_elapsed_max = max(completed_elapsed_max or 0., elapsed_request)
            else:
                survivors.append(ServingRequest(request.request_id, request.input_tokens,
                    request.output_tokens, request.generated_decode_tokens + 1,
                    request.prefill_first_output_token, request.slot_id))
        active = survivors
        if not measuring:
            warmup_completed += 1
            continue
        measurement_end = now
        measured += 1
        tokens += step_tokens
        completed += step_completed
        admitted += len(iteration_admissions)
        full += step_tokens == config.batch_size
        batch_sum += step_tokens
        total_intervals += sum(intervals)
        max_interval = max(max_interval, max(intervals))
        max_iteration = max(max_iteration, now - iteration_start)
        stall += decode_start - iteration_start
        for imported in iteration_admissions:
            pd_bytes += imported.transfer_bytes
            pd_program += imported.physical_program_bytes
        decode_program += execution.physical_program_bytes
        logical_write += execution.logical_write_bytes
        transfer += execution.transfer_bytes
        add_costs(resource_costs, execution.costs)
        if config.retain_step_records:
            records.append({"step": measured, "start_ns": iteration_start,
                "decode_start_ns": decode_start, "end_ns": now,
                "batch_size": step_tokens, "admitted": len(iteration_admissions),
                "completed": step_completed, "pending": len(pending),
                "min_tpot_ns": min(intervals), "max_tpot_ns": max(intervals)})
    elapsed = max(0., measurement_end - measurement_start)
    seconds = elapsed / NS_PER_SECOND
    unfinished_transfer_end = sum(row.transfer_bytes for row in pending if row.ready_ns > measurement_end)
    unfinished_program_end = sum(row.physical_program_bytes for row in pending if row.ready_ns > measurement_end)
    # Completion conservation avoids attributing unfinished final prefetches to
    # the window, or charging old, already-ready KV only when it is admitted.
    transfer_completed = pd_transfer_issued + unfinished_transfer_start - unfinished_transfer_end
    program_completed = pd_program_issued + unfinished_program_start - unfinished_program_end
    pd_costs_completed = dict(pd_costs_issued)
    add_costs(pd_costs_completed, unfinished_costs_start)
    for row in pending:
        if row.ready_ns > measurement_end:
            for name, value in row.costs.items():
                pd_costs_completed[name] = pd_costs_completed.get(name, 0.) - value
    if not measuring_imports:
        transfer_completed = program_completed = 0
        unfinished_transfer_start = unfinished_program_start = 0
        pd_costs_completed.clear()
    add_costs(resource_costs, pd_costs_completed)
    physical = program_completed + decode_program
    projected = physical / seconds * THREE_YEARS_SECONDS if seconds else 0.
    return ServingResult(status=status, config=asdict(config), startup_end_ns=startup_end,
        startup_import_bytes=startup_bytes, startup_program_bytes=startup_program,
        startup_costs_ns=startup_costs, measurement_start_ns=measurement_start,
        measurement_end_ns=measurement_end, warmup_steps_completed=warmup_completed,
        measured_steps=measured, decode_tokens=tokens, completed_requests=completed,
        completed_request_elapsed_count=completed_elapsed_count,
        mean_completed_request_elapsed_ns=(completed_elapsed_sum / completed_elapsed_count
                                          if completed_elapsed_count else None),
        max_completed_request_elapsed_ns=completed_elapsed_max,
        admitted_requests=admitted, decode_tokens_per_second=tokens / seconds if seconds else 0.,
        output_equivalent_tokens_per_second=(completed * config.output_tokens / seconds if seconds else 0.),
        mean_tpot_ns=total_intervals / tokens if tokens else 0., max_tpot_ns=max_interval,
        mean_iteration_ns=elapsed / measured if measured else 0., max_iteration_ns=max_iteration,
        mean_batch_size=batch_sum / measured if measured else 0.,
        full_batch_step_fraction=full / measured if measured else 0., admission_stall_ns=stall,
        pd_import_bytes=pd_bytes, pd_program_bytes=pd_program,
        pd_transfer_issued_bytes=pd_transfer_issued, pd_program_issued_bytes=pd_program_issued,
        pd_transfer_completed_bytes=transfer_completed, pd_program_completed_bytes=program_completed,
        pending_unfinished_transfer_start_bytes=unfinished_transfer_start,
        pending_unfinished_program_start_bytes=unfinished_program_start,
        pending_unfinished_transfer_end_bytes=unfinished_transfer_end,
        pending_unfinished_program_end_bytes=unfinished_program_end,
        decode_program_bytes=decode_program, logical_decode_write_bytes=logical_write,
        decode_transfer_bytes=transfer, pd_resource_service_ns=pd_costs_completed,
        resource_service_ns=resource_costs,
        projected_three_year_physical_bytes=projected,
        projected_three_year_average_pec=(projected / config.hbf_capacity_bytes
                                        if config.hbf_capacity_bytes else None),
        pending_at_measurement_start=pending_start, pending_at_measurement_end=len(pending),
        step_records=records, assumptions=[
            "Explicit complete decode events; no cycle extrapolation or skipped tokens.",
            "Warmup and measurement are complete-event windows; startup PD imports are separate.",
            "Callbacks own shared resource calendars and prepared-KV capacity; admission requires ready time.",
            "Every request and prepared successor owns a stable slot_id; replacement never migrates "
            "KV by changing positions in the active list.",
            "PD admission, issue and whole-import completion bytes are separate. PEC uses completed "
            "program bytes, with unfinished start/end imports corrected by byte conservation; "
            "individual sub-program completion times remain callback-specific.",
            "Service occupancies can overlap and must not be added to infer wall-clock time.",
            "Mean request TPOT excludes its own PD admission delay; mean_iteration_ns includes all "
            "admission stalls and is the full-batch runtime/SLO comparison metric.",
            "Completed-request residence measures fresh admission through last decode output, including "
            "intervening stalls; completions inside measurement may have begun in warmup. Staggered "
            "startup seeds are excluded. External prefill/PD-before-admission delay is not included.",
            "Three-year PEC is physical programmed bytes / usable HBF bytes under ideal uniform wear; "
            "it is not measured erase counts, an FTL/GC prediction, or a NAND endurance guarantee.",
            "Observed closed-loop window is not a proof of a periodic steady state."])
