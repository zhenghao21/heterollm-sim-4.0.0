"""Records of the native stationary fast mode.

The live runtime stops at the first real cohort boundary. A window forecast
is a separate value; neither the native requests nor any physical calendar
advance through its predicted window. Layer samples are complete physical
tasks on independent copies, never completed cohorts or visible Tokens.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class NativeWindowForecast:
    cutoff_ns: float
    sampled_decode_cohorts: int
    predicted_decode_cohorts: int
    sampled_visible_output_tokens: int
    predicted_visible_output_tokens: int
    original_window: Mapping[str, Any]
    measurement_window: Mapping[str, Any]
    metrics: Mapping[str, Any]
    metric_bounds: Mapping[str, Any]
    request_forecasts: tuple[Mapping[str, Any], ...]
    error_budget: Mapping[str, Any]

    def __post_init__(self):
        if self.sampled_decode_cohorts or self.sampled_visible_output_tokens:
            raise ValueError("native layer samples cannot claim completed Decode cohorts or visible output Tokens")

    def to_metadata(self) -> dict:
        return {"schema": "native-stationary-window-forecast/v1", **asdict(self),
                "controller_state": "discarded_after_independent_native_layer_samples",
                "forecast_has_physical_trace": False}


_SAMPLE_ROLES = frozenset(("calibration", "holdout", "entry", "exit", "pd_import", "prefill_pass"))


@dataclass(frozen=True)
class NativeLayerWindowSample:
    """One completely executed native sample run, not a complete Token.

    Times and transaction observations refer only to these real tasks. Any
    imported KV or boundary initialization is accounted separately; no
    request/cohort completion can be inferred from this record.
    """
    window_id: str
    role: str
    cohort_id: str
    cohort_kind: str
    decode_step: int
    layer_ids: tuple[str, ...]
    rank_ids: tuple[int, ...]
    context_tokens_by_request: Mapping[str, int]
    executed_task_count: int
    start_ns: float
    end_ns: float
    wall_seconds: float
    physical_observations: Mapping[str, Any]

    def __post_init__(self):
        if (type(self.window_id) is not str or not self.window_id
                or type(self.cohort_id) is not str or not self.cohort_id
                or type(self.role) is not str or self.role not in _SAMPLE_ROLES):
            raise ValueError("native layer sample requires an identity and a stationary sample role")
        if type(self.cohort_kind) is not str or self.cohort_kind not in {"prefill", "decode"}:
            raise ValueError("native layer sample requires an ordinary prefill or decode source cohort")
        if type(self.decode_step) is not int or self.decode_step < 0:
            raise ValueError("native layer sample decode_step must be a nonnegative integer")
        if (not isinstance(self.layer_ids, tuple) or not self.layer_ids
                or any(type(value) is not str or not value for value in self.layer_ids)
                or len(set(self.layer_ids)) != len(self.layer_ids)):
            raise ValueError("native layer sample requires distinct native layer ids")
        if (not isinstance(self.rank_ids, tuple) or not self.rank_ids
                or any(type(value) is not int or value < 0 for value in self.rank_ids)
                or len(set(self.rank_ids)) != len(self.rank_ids)):
            raise ValueError("native layer sample requires distinct nonnegative rank ids")
        if (not isinstance(self.context_tokens_by_request, Mapping) or not self.context_tokens_by_request
                or any(type(key) is not str or not key or type(value) is not int or value < 0
                       for key, value in self.context_tokens_by_request.items())):
            raise ValueError("native layer sample requires explicit request context positions")
        if type(self.executed_task_count) is not int or self.executed_task_count < 1:
            raise ValueError("native layer sample requires a positive executed task count")
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in (self.start_ns, self.end_ns, self.wall_seconds))
                or self.start_ns < 0 or self.end_ns < self.start_ns or self.wall_seconds < 0):
            raise ValueError("native layer sample requires finite ordered physical times and wall duration")
        if not isinstance(self.physical_observations, Mapping):
            raise ValueError("native layer sample observations must be a mapping")

    def to_metadata(self):
        return {"schema": "native-physical-layer-window-sample/v1", **asdict(self),
            "window_fully_executed": True, "complete_native_cohort": False,
            "visible_output_tokens_observed": 0,
            "state_scope": "boundary_conditioned_native_layer_run_only",
            "physical_trace_scope": "only_executed_window_tasks"}


def _eligibility_reasons(scenario, plan, *, custom_lowerer, shared_residency):
    reasons = []
    ingress = scenario.workload.metadata.get("external_kv_ingress")
    if not isinstance(ingress, Mapping):
        reasons.append("native_window_requires_external_prefill_ingress")
    elif (type(ingress.get("warmup_decode_cohorts")) is not int
            or type(ingress.get("measurement_decode_cohorts")) is not int
            or ingress["warmup_decode_cohorts"] < 0 or ingress["measurement_decode_cohorts"] < 1):
        reasons.append("native_window_requires_explicit_positive_measurement_window")
    return (*reasons, *_native_runtime_reasons(plan, custom_lowerer=custom_lowerer,
                                               shared_residency=shared_residency))


def _native_runtime_reasons(plan, *, custom_lowerer, shared_residency):
    """Runtime features no native layer-sample composition bounds."""
    reasons = []
    if custom_lowerer:
        reasons.append("native_window_requires_standard_native_lowerer")
    if shared_residency:
        reasons.append("native_window_shared_residency_not_bounded")
    if plan.mtp.enabled:
        reasons.append("native_window_speculative_decode_not_bounded")
    if plan.scheduler.preemption_enabled:
        reasons.append("native_window_preemption_not_bounded")
    if plan.kv_policy.offload_component or plan.kv_policy.prefetch_distance:
        reasons.append("native_window_offload_or_prefetch_not_bounded")
    return tuple(reasons)


def _checked_budget(budget):
    if not isinstance(budget, Mapping):
        raise TypeError("native window budget must return a mapping")
    if budget.get("accepted") is not True:
        return budget
    invalid = []
    metrics = budget.get("metrics")
    if not isinstance(metrics, Mapping):
        invalid.append("native_window_accepted_budget_metrics_missing")
    else:
        for name in ("cluster_tokens_per_second", "tokens_per_second_per_gpu",
                     "mean_complete_request_tpot_ns", "mean_complete_request_ingress_ttft_ns",
                     "measurement_duration_ns"):
            value = metrics.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                invalid.append("native_window_accepted_budget_invalid_" + name)
        for name in ("complete_fresh_requests", "measurement_visible_output_tokens"):
            if type(metrics.get(name)) is not int or metrics[name] < 1:
                invalid.append("native_window_accepted_budget_invalid_" + name)
    predicted = budget.get("predicted_visible_output_tokens")
    if type(predicted) is not int or predicted < 0:
        invalid.append("native_window_accepted_budget_invalid_predicted_visible_output_tokens")
    for name in ("window", "metric_bounds", "error_budget"):
        if not isinstance(budget.get(name), Mapping):
            invalid.append("native_window_accepted_budget_missing_" + name)
    if invalid:
        return {**budget, "accepted": False, "can_expand_sampling": False,
                "reasons": [*budget.get("reasons", ()), *invalid]}
    return budget
