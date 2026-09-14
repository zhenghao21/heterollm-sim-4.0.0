"""Thread-safe background simulation jobs with bounded terminal history."""

from __future__ import annotations

import copy
import math
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from .config import ScenarioConfig
from .contracts import RetentionPolicy
from .execution_control import (
    ExecutionCancelledError,
    ExecutionControl,
    ExecutionProgress,
    progress_unit_for_stage,
    progress_unit_label_zh,
)
from .reporting import (
    OnlineScenarioResult,
    RunResult,
    page_online_batch_trace,
    report_dict,
    replay_online_batch_trace,
    run_scenario,
)
from .run_estimation import estimate_scenario


QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATUSES = frozenset({COMPLETED, FAILED, CANCELLED})
ALL_STATUSES = frozenset({QUEUED, RUNNING}) | TERMINAL_STATUSES


class RunJobCapacityError(RuntimeError):
    """Raised when the bounded active-job queue is full."""


class RunJobTraceError(RuntimeError):
    """A safe, client-facing trace lookup failure."""

    def __init__(
        self,
        status: int,
        code: str,
        message_zh: str,
        *,
        message_en: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message_zh)
        self.status = int(status)
        self.code = str(code)
        self.message_zh = str(message_zh)
        self.message_en = str(message_en)
        self.details = dict(details or {})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    """Copy ``value`` into JSON-compatible standard-library containers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    # Progress metadata is extension-facing.  A stable string fallback keeps a
    # snapshot serializable without exposing object internals or repr addresses.
    return type(value).__name__


def _progress_dict(
    stage: str,
    completed: int,
    total: Optional[int],
    message: str,
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    unit: Optional[str] = None,
    scope: str = "primary",
) -> Dict[str, Any]:
    completed_value = max(0, int(completed))
    total_value = None if total is None else max(0, int(total))
    ratio: Optional[float]
    if total_value is None or total_value == 0:
        ratio = None
    else:
        ratio = min(1.0, max(0.0, completed_value / float(total_value)))
    logical_unit = str(unit or progress_unit_for_stage(stage))
    return {
        "stage": str(stage),
        "completed": completed_value,
        "total": total_value,
        "ratio": ratio,
        "unit": logical_unit,
        "unit_label_zh": progress_unit_label_zh(logical_unit),
        "scope": str(scope),
        "message": str(message),
        "metadata": _json_safe(dict(metadata or {})),
    }


@dataclass
class _RunJob:
    job_id: str
    scenario: ScenarioConfig
    retention_policy: str
    estimate: Dict[str, Any]
    created_at: str
    status: str = QUEUED
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: Dict[str, Any] = field(
        default_factory=lambda: _progress_dict(
            QUEUED, 0, 1, "仿真任务已排队"
        )
    )
    progress_detail: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    result: Optional[RunResult] = None
    error: Optional[Dict[str, str]] = None
    cancellation_requested: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Optional[Future] = None
    finished_sequence: Optional[int] = None


@dataclass
class _TraceReplayFlight:
    event: threading.Event = field(default_factory=threading.Event)
    replay: Optional[Dict[str, Any]] = None
    error: Optional[BaseException] = None


class RunJobManager:
    """Execute simulations on a finite worker pool.

    Active jobs are never evicted.  Once a job becomes terminal, only the
    newest ``max_history`` jobs by terminal-completion order are retained.
    Their full results remain private and available for lazy batch replay
    until that bounded history evicts or explicitly prunes the job.
    """

    def __init__(
        self,
        max_workers: int = 2,
        max_history: int = 100,
        max_active_jobs: int = 8,
        max_trace_cache_entries: int = 8,
    ) -> None:
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers < 1
        ):
            raise ValueError("max_workers 必须是正整数")
        if (
            isinstance(max_history, bool)
            or not isinstance(max_history, int)
            or max_history < 1
        ):
            raise ValueError("max_history 必须是正整数")
        if (
            isinstance(max_active_jobs, bool)
            or not isinstance(max_active_jobs, int)
            or max_active_jobs < max_workers
        ):
            raise ValueError("max_active_jobs 必须是不小于 max_workers 的正整数")
        if (
            isinstance(max_trace_cache_entries, bool)
            or not isinstance(max_trace_cache_entries, int)
            or max_trace_cache_entries < 1
        ):
            raise ValueError("max_trace_cache_entries 必须是正整数")
        self._max_history = max_history
        self._max_active_jobs = max_active_jobs
        self._max_trace_cache_entries = max_trace_cache_entries
        self._lock = threading.RLock()
        self._jobs: "OrderedDict[str, _RunJob]" = OrderedDict()
        self._trace_cache: "OrderedDict[tuple[str, str], Dict[str, Any]]" = (
            OrderedDict()
        )
        self._trace_inflight: Dict[tuple[str, str], _TraceReplayFlight] = {}
        self._finished_sequence = 0
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="heterollm-run",
        )
        self._shutdown = False

    @property
    def max_history(self) -> int:
        return self._max_history

    @property
    def max_active_jobs(self) -> int:
        return self._max_active_jobs

    @property
    def max_trace_cache_entries(self) -> int:
        return self._max_trace_cache_entries

    @property
    def trace_cache_size(self) -> int:
        with self._lock:
            return len(self._trace_cache)

    def submit(
        self, scenario: ScenarioConfig, retention_policy: Optional[str] = None
    ) -> str:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError("scenario 必须是 ScenarioConfig")
        if retention_policy is not None and not isinstance(retention_policy, str):
            raise TypeError(
                "retention_policy 必须是 exact、streaming 或 aggregate"
            )
        selected_policy: Optional[str] = None
        if retention_policy is not None:
            try:
                selected_policy = RetentionPolicy(retention_policy).value
            except ValueError as exc:
                raise ValueError(
                    "retention_policy 必须是 exact、streaming 或 aggregate"
                ) from exc
        with self._lock:
            if self._shutdown:
                raise RuntimeError("RunJobManager 已关闭")
            self._ensure_submission_capacity_locked()
        estimate = estimate_scenario(scenario)
        if selected_policy is None:
            selected_policy = RetentionPolicy(
                str(estimate["recommended_retention_policy"])
            ).value
        scheduler = getattr(scenario.workload, "scheduler", None)
        scheduler_mode = (
            str(getattr(scheduler, "mode", "static")) if scheduler else "static"
        )
        if scheduler_mode == "continuous" and selected_policy != "aggregate":
            raise ValueError(
                "continuous 调度仅支持 aggregate retention_policy；"
                "批次外层不会伪装成 exact 或 streaming 任务轨迹"
            )
        job_id = uuid.uuid4().hex
        job = _RunJob(
            job_id=job_id,
            scenario=scenario,
            retention_policy=selected_policy,
            estimate=estimate,
            created_at=_now(),
        )
        with self._lock:
            if self._shutdown:
                raise RuntimeError("RunJobManager 已关闭")
            # Re-check after estimation so concurrent submitters cannot exceed
            # the bound.  The first check above keeps a full queue O(1).
            self._ensure_submission_capacity_locked()
            self._jobs[job_id] = job
            try:
                job.future = self._executor.submit(self._run_job, job_id)
            except Exception:
                del self._jobs[job_id]
                raise
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(str(job_id))
            return None if job is None else self._snapshot_locked(job)

    def trace_page(
        self,
        job_id: str,
        batch_id: str,
        *,
        offset: int = 0,
        limit: int = 5_000,
    ) -> Dict[str, Any]:
        """Replay a page of exact task events for one completed online batch."""

        if not isinstance(batch_id, str) or not batch_id:
            raise RunJobTraceError(
                400,
                "invalid_batch_id",
                "batch_id 必须是非空字符串",
                message_en="batch_id must be a non-empty string",
            )
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise RunJobTraceError(
                400,
                "invalid_trace_offset",
                "offset 必须是非负整数",
                message_en="offset must be a non-negative integer",
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > 5_000
        ):
            raise RunJobTraceError(
                400,
                "invalid_trace_limit",
                "limit 必须是 1 到 5000 之间的整数",
                message_en="limit must be an integer between 1 and 5000",
            )

        cache_key = (str(job_id), batch_id)
        owner = False
        with self._lock:
            result = self._trace_result_locked(cache_key[0])
            replay = self._trace_cache.get(cache_key)
            if replay is not None:
                self._trace_cache.move_to_end(cache_key)
                flight = None
            else:
                flight = self._trace_inflight.get(cache_key)
                if flight is None:
                    flight = _TraceReplayFlight()
                    self._trace_inflight[cache_key] = flight
                    owner = True

        if replay is None and not owner:
            # A concurrent caller owns the expensive replay.  Waiting never
            # holds the manager lock, so job progress and pruning stay live.
            assert flight is not None
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            replay = flight.replay
            if replay is None:  # defensive: every owner signals one outcome
                raise RuntimeError("批次时间轴缓存构建未返回结果")

        if replay is None:
            assert owner and flight is not None
            build_error: Optional[BaseException] = None
            try:
                replay = replay_online_batch_trace(result, batch_id)
            except KeyError as exc:
                build_error = RunJobTraceError(
                    404,
                    "run_job_batch_not_found",
                    "未找到指定的在线批次",
                    message_en="online batch not found",
                    details={"batch_id": batch_id},
                )
                build_error.__cause__ = exc
            except BaseException as exc:  # wake every waiter before re-raising
                build_error = exc

            with self._lock:
                current = self._jobs.get(cache_key[0])
                if build_error is None and (
                    current is None
                    or current.status != COMPLETED
                    or current.result is not result
                ):
                    build_error = RunJobTraceError(
                        404,
                        "run_job_not_found",
                        "未找到指定的仿真任务",
                        message_en="simulation job not found",
                    )
                if build_error is None and replay is not None:
                    if not self._shutdown:
                        self._trace_cache[cache_key] = replay
                        self._trace_cache.move_to_end(cache_key)
                        while (
                            len(self._trace_cache)
                            > self._max_trace_cache_entries
                        ):
                            self._trace_cache.popitem(last=False)
                    flight.replay = replay
                else:
                    flight.error = build_error
                if self._trace_inflight.get(cache_key) is flight:
                    del self._trace_inflight[cache_key]
                flight.event.set()
            if build_error is not None:
                raise build_error

        # Cache values are manager-owned; callers receive an isolated page so
        # direct Python consumers cannot mutate a later HTTP response.
        return copy.deepcopy(
            page_online_batch_trace(replay, offset=offset, limit=limit)
        )

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None or job.status in TERMINAL_STATUSES:
                return False
            job.cancellation_requested = True
            job.cancel_event.set()
            if job.status == QUEUED:
                if job.future is not None:
                    job.future.cancel()
                self._mark_cancelled_locked(job)
                self._prune_terminal_locked(self._max_history)
            else:
                job.progress = _progress_dict(
                    "cancelling",
                    job.progress["completed"],
                    job.progress["total"],
                    "正在安全取消仿真任务",
                    job.progress.get("metadata"),
                    unit=job.progress.get("unit"),
                )
            return True

    def list(
        self,
        status: Optional[str] = None,
        *,
        statuses: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        selected: Optional[Set[str]] = None
        if status is not None and statuses is not None:
            raise ValueError("status 与 statuses 不能同时使用")
        if status is not None:
            selected = {str(status)}
        elif statuses is not None:
            selected = {str(item) for item in statuses}
        if selected is not None and not selected.issubset(ALL_STATUSES):
            raise ValueError("包含未知的 Job 状态")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError("limit 必须是非负整数或 None")
        with self._lock:
            jobs = reversed(tuple(self._jobs.values()))
            snapshots = [
                self._snapshot_locked(job)
                for job in jobs
                if selected is None or job.status in selected
            ]
        return snapshots if limit is None else snapshots[:limit]

    def prune(
        self,
        job_id: Optional[str] = None,
        *,
        keep: Optional[int] = None,
    ) -> int:
        """Remove terminal jobs.

        With ``job_id``, remove that terminal job only.  With ``keep``, retain
        the newest N terminal jobs.  With neither argument, remove all terminal
        jobs.  Queued and running jobs are always preserved.
        """

        if job_id is not None and keep is not None:
            raise ValueError("job_id 与 keep 不能同时使用")
        if keep is not None and (
            isinstance(keep, bool) or not isinstance(keep, int) or keep < 0
        ):
            raise ValueError("keep 必须是非负整数或 None")
        with self._lock:
            if job_id is not None:
                job = self._jobs.get(str(job_id))
                if job is None or job.status not in TERMINAL_STATUSES:
                    return 0
                self._remove_job_locked(job.job_id)
                return 1
            return self._prune_terminal_locked(0 if keep is None else keep)

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_pending: bool = False,
        cancel_running: bool = False,
    ) -> None:
        with self._lock:
            self._shutdown = True
            self._trace_cache.clear()
            if cancel_pending or cancel_running:
                targets = [
                    job.job_id
                    for job in self._jobs.values()
                    if (cancel_pending and job.status == QUEUED)
                    or (cancel_running and job.status == RUNNING)
                ]
            else:
                targets = []
        for job_id in targets:
            self.cancel(job_id)
        self._executor.shutdown(wait=wait, cancel_futures=cancel_pending)

    def __enter__(self) -> "RunJobManager":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.shutdown(wait=True)

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status == CANCELLED:
                return
            job.status = RUNNING
            job.started_at = _now()
            job.progress = _progress_dict(
                "starting", 0, 1, "正在启动仿真任务"
            )

        control = ExecutionControl(
            progress_callback=lambda progress: self._update_progress(job_id, progress),
            cancellation_callback=job.cancel_event.is_set,
        )
        try:
            result = run_scenario(
                job.scenario,
                retention_policy=job.retention_policy,
                control=control,
            )
            control.raise_if_cancelled()
            report = _json_safe(report_dict(result))
            control.raise_if_cancelled()
        except ExecutionCancelledError:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None and current.status not in TERMINAL_STATUSES:
                    self._mark_cancelled_locked(current)
                    self._prune_terminal_locked(self._max_history)
            return
        except Exception as exc:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None and current.status not in TERMINAL_STATUSES:
                    if current.cancel_event.is_set():
                        self._mark_cancelled_locked(current)
                    else:
                        current.status = FAILED
                        current.finished_at = _now()
                        self._record_terminal_locked(current)
                        current.error = {
                            "message": "仿真任务失败，请检查场景配置后重试。",
                            "exception_type": type(exc).__name__,
                        }
                        current.progress = _progress_dict(
                            FAILED,
                            current.progress["completed"],
                            current.progress["total"],
                            "仿真任务执行失败",
                            current.progress.get("metadata"),
                            unit=current.progress.get("unit"),
                        )
                        current.progress_detail = None
                    self._prune_terminal_locked(self._max_history)
            return

        with self._lock:
            current = self._jobs.get(job_id)
            if current is None or current.status in TERMINAL_STATUSES:
                return
            if current.cancel_event.is_set():
                self._mark_cancelled_locked(current)
            else:
                current.report = report
                current.result = result
                current.status = COMPLETED
                current.finished_at = _now()
                self._record_terminal_locked(current)
                current.progress = _progress_dict(
                    COMPLETED, 1, 1, "仿真任务已完成"
                )
                current.progress_detail = None
            self._prune_terminal_locked(self._max_history)

    def _trace_result_locked(self, job_id: str) -> OnlineScenarioResult:
        job = self._jobs.get(job_id)
        if job is None:
            raise RunJobTraceError(
                404,
                "run_job_not_found",
                "未找到指定的仿真任务",
                message_en="simulation job not found",
            )
        if job.status != COMPLETED:
            raise RunJobTraceError(
                409,
                "run_job_trace_not_ready",
                "仅已完成的仿真任务可以读取批次时间轴",
                message_en="batch traces are available only for completed jobs",
                details={"status": job.status},
            )
        if not isinstance(job.result, OnlineScenarioResult):
            raise RunJobTraceError(
                409,
                "run_job_trace_unavailable",
                "该仿真任务不是在线连续批处理结果，无法读取批次时间轴",
                message_en=(
                    "the job is not an online continuous-batching result"
                ),
            )
        return job.result

    def _ensure_submission_capacity_locked(self) -> None:
        active_count = sum(
            job.status not in TERMINAL_STATUSES for job in self._jobs.values()
        )
        if active_count >= self._max_active_jobs:
            raise RunJobCapacityError(
                "后台仿真队列已满；请等待现有任务完成或先取消不再需要的任务"
            )

    def _update_progress(self, job_id: str, progress: ExecutionProgress) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if (
                job is None
                or job.status != RUNNING
                or job.cancel_event.is_set()
            ):
                return
            metadata = dict(progress.metadata)
            if progress.simulated_time_ns is not None:
                metadata["simulated_time_ns"] = progress.simulated_time_ns
            update = _progress_dict(
                progress.stage,
                progress.completed,
                progress.total,
                progress.message,
                metadata,
                unit=progress.unit,
            )
            if (
                progress.stage == "cohort_tasks"
                and job.progress.get("unit") == "serving_batches"
            ):
                update["scope"] = "nested"
                job.progress_detail = update
                return
            job.progress = update
            if progress.stage not in {"serving_cohorts", "serving_complete"}:
                job.progress_detail = None

    def _mark_cancelled_locked(self, job: _RunJob) -> None:
        job.status = CANCELLED
        job.finished_at = _now()
        self._record_terminal_locked(job)
        job.report = None
        job.result = None
        job.error = None
        job.progress = _progress_dict(
            CANCELLED,
            job.progress["completed"],
            job.progress["total"],
            "仿真任务已取消",
            job.progress.get("metadata"),
            unit=job.progress.get("unit"),
        )
        job.progress_detail = None

    def _prune_terminal_locked(self, keep: int) -> int:
        terminal_jobs = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status in TERMINAL_STATUSES
            ),
            key=lambda job: (
                job.finished_sequence
                if job.finished_sequence is not None
                else -1,
                job.finished_at or "",
                job.job_id,
            ),
        )
        remove_count = max(0, len(terminal_jobs) - keep)
        for job in terminal_jobs[:remove_count]:
            self._remove_job_locked(job.job_id)
        return remove_count

    def _record_terminal_locked(self, job: _RunJob) -> None:
        if job.finished_sequence is None:
            self._finished_sequence += 1
            job.finished_sequence = self._finished_sequence

    def _remove_job_locked(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        for cache_key in tuple(self._trace_cache):
            if cache_key[0] == job_id:
                del self._trace_cache[cache_key]

    @staticmethod
    def _snapshot_locked(job: _RunJob) -> Dict[str, Any]:
        progress = copy.deepcopy(job.progress)
        if job.progress_detail is not None:
            progress["detail"] = copy.deepcopy(job.progress_detail)
        snapshot = {
            "job_id": job.job_id,
            "status": job.status,
            "retention_policy": job.retention_policy,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "cancellation_requested": job.cancellation_requested,
            "progress": progress,
            "estimate": job.estimate,
            "report": job.report,
            "error": job.error,
        }
        return copy.deepcopy(snapshot)


__all__ = [
    "ALL_STATUSES",
    "CANCELLED",
    "COMPLETED",
    "FAILED",
    "QUEUED",
    "RUNNING",
    "RunJobManager",
    "RunJobCapacityError",
    "RunJobTraceError",
    "TERMINAL_STATUSES",
]
