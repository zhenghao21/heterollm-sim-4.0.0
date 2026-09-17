"""Unpriced CPU graph lifecycle diagnostics for realized llama serving cohorts.

The recorder belongs to one _OnlineRuntime/llama_context, never to a cost
provider or a template cache. It consumes already-lowered physical ubatches.
Source-derived adapter contracts qualify only the fields stated below; no file
or binary verification is performed here and native execution is not observed.
"""
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from .llama_graph_lifecycle import (
    GraphParameters, GraphState, execution_failed, process_ubatch,
    CONTEXT_SOURCE, PARAM_SOURCE, INPUT_SOURCE,
)

SCHEMA = "heterollm.llama-cpu-graph-runtime/v1"
RECORD_LIMIT = 4096
ORDINARY_GROUPS = {"stateless_scheduler_batch", "explicit_mixed_phase_physical_batch"}


@dataclass(frozen=True)
class PreparedGraphCohort:
    revision: int
    cohort_id: str
    state: GraphState
    records: Tuple[Mapping[str, object], ...] = ()
    unobserved_reason: Optional[str] = None


class LlamaGraphRuntime:
    """Prepare pure transitions, commit only after the real cohort succeeds."""

    def __init__(self, scenario):
        self.state = GraphState.unknown()
        self.revision = 0
        self.successful_cohorts = 0
        self.observed_cohorts = 0
        self.failed_cohorts = 0
        self.unobserved_reasons = Counter()
        self.decisions = Counter()
        self.reasons = Counter()
        self.records = []
        self.failure_records = []
        self.dropped_records = 0
        self.config = getattr(scenario, "llama_cpp_config", None)
        metadata = getattr(getattr(scenario, "workload", None), "metadata", {})
        metadata = metadata if isinstance(metadata, Mapping) else {}
        contract = metadata.get("llama_cpp_mixed_phase_batching", {})
        contract = contract if isinstance(contract, Mapping) else {}
        self.contract = {key: contract.get(key) for key in
                         ("schema", "status", "reason", "source", "evidence_kind")}
        self.qualified = (
            self.config is not None
            and contract.get("schema") == "llama.cpp.mixed-phase-batching/v1"
            and contract.get("graph_qualified") is True
            and contract.get("evidence_kind") == "source_derived_execution_semantics"
            and getattr(getattr(scenario, "workload", None), "mtp", None) is None
        )
        self.simple = self.qualified and contract.get("reason") == "ordinary_dense_full_attention_text" and self.config.kv_unified is True
        self.qualification = {
            "scope": "existing_source_derived_llama_batching_contract" if self.qualified else "unobserved",
            "adapter_contract": self.contract,
            "source_identity_qualification": "partial_adapter_contract_only_no_graph_source_or_binary_reverification",
            "source_references": (CONTEXT_SOURCE, PARAM_SOURCE, INPUT_SOURCE),
            "native_execution_verified": False,
            "initial_graph": "unknown_after_warmup_or_unobserved_initialization",
            "complete_input_roster": "unknown",
            "exact_padded_kv_view": "unknown_lower_bounds_are_not_equalities",
        }

    def prepare(self, cohort, metadata) -> PreparedGraphCohort:
        ident = str(getattr(cohort, "cohort_id", ""))
        def unobserved(reason):
            return PreparedGraphCohort(self.revision, ident,
                GraphState.unknown(generation=self.state.generation), unobserved_reason=reason)
        if not self.qualified:
            return unobserved("llama_source_derived_batching_contract_unavailable")
        if getattr(cohort, "kind", None) not in {"prefill", "decode", "mixed"}:
            return unobserved("cohort_outside_ordinary_prefill_decode_scope")
        if not isinstance(metadata, Mapping):
            return unobserved("cost_metadata_unavailable")
        groups = metadata.get("operator_invocation_groups")
        count = metadata.get("operator_invocation_group_count")
        if (not isinstance(groups, (tuple, list)) or not groups or type(count) is not int
                or count != len(groups)):
            return unobserved("complete_physical_ubatch_groups_unavailable")
        state = self.state
        records = []
        total_tokens = 0
        group_ids = set()
        for index, group in enumerate(groups):
            if not isinstance(group, Mapping):
                return unobserved("invalid_physical_ubatch_group")
            n = group.get("token_batch")
            outputs = group.get("logit_token_batch")
            lanes = group.get("lanes")
            requests = group.get("request_ids")
            group_id = group.get("group_id")
            if (group.get("group_index") != index or type(group.get("group_index")) is not int
                    or not isinstance(group_id, str) or not group_id or group_id in group_ids
                    or type(n) is not int or n <= 0 or group.get("physical_ubatch_rows") != n
                    or type(outputs) is not int or not 0 <= outputs <= n
                    or not isinstance(lanes, (tuple, list)) or len(lanes) != n
                    or not isinstance(requests, (tuple, list)) or not requests
                    or any(not isinstance(r, str) or not r for r in requests)
                    or len(set(requests)) != len(requests)):
                return unobserved("invalid_physical_ubatch_identity_or_shape")
            if any(not isinstance(lane, Mapping) or lane.get("request_id") not in requests
                   or type(lane.get("requires_logits")) is not bool for lane in lanes):
                return unobserved("incomplete_physical_ubatch_lanes")
            if (set(requests) != {lane["request_id"] for lane in lanes}
                    or outputs != sum(lane["requires_logits"] for lane in lanes)):
                return unobserved("physical_ubatch_output_or_participant_mismatch")
            total_tokens += n
            group_ids.add(group_id)
            # Only split_simple is identified here. Equal-sized requests alone
            # do not establish the native equal_seqs path or native seq IDs.
            simple = self.simple and group.get("batching_semantics") in ORDINARY_GROUPS
            params = GraphParameters(
                n_tokens=n, n_outputs=outputs,
                equal_seqs=False if simple else None,
                n_seqs=n if simple else None,
                n_seq_tokens=1 if simple else None,
            )
            transition = process_ubatch(state, params, input_roster=None,
                disable_reuse=None, pipeline_parallel=None, buffer_expansion="unknown")
            records.append({
                "cohort_id": ident, "group_id": group_id, "group_index": index,
                "context_id": "llama_context:main", "request_ids": tuple(requests),
                "batching_semantics": group.get("batching_semantics"),
                "parameters_known": {key: value for key, value in asdict(params).items() if value is not None},
                "output_row_indices": tuple(i for i, lane in enumerate(lanes) if lane["requires_logits"]),
                "decision": transition.decision, "reasons": transition.reasons,
                "events": tuple({"kind": event.kind, "evidence": event.evidence} for event in transition.events),
                "generation_before": state.generation, "generation_after": transition.state.generation,
                "native_execution_verified": False, "priced": False,
            })
            state = transition.state
        cohort_tokens = getattr(cohort, "token_count", None)
        if type(cohort_tokens) is not int or total_tokens != cohort_tokens:
            return unobserved("physical_ubatch_token_coverage_mismatch")
        return PreparedGraphCohort(self.revision, ident, state, tuple(records))

    def commit(self, prepared: PreparedGraphCohort) -> None:
        if prepared.revision != self.revision:
            raise ValueError("graph lifecycle transaction is stale or already committed")
        self.state = prepared.state
        self.revision += 1
        self.successful_cohorts += 1
        if prepared.unobserved_reason:
            self.unobserved_reasons[prepared.unobserved_reason] += 1
            return
        self.observed_cohorts += 1
        for record in prepared.records:
            self.decisions[record["decision"]] += 1
            self.reasons.update(record["reasons"])
            if len(self.records) < RECORD_LIMIT:
                self.records.append(record)
            else:
                self.dropped_records += 1

    def failed(self, cohort_id: str, error: BaseException) -> None:
        transition = execution_failed(self.state, reason=type(error).__name__)
        self.state = transition.state
        self.revision += 1
        self.failed_cohorts += 1
        if len(self.failure_records) < RECORD_LIMIT:
            self.failure_records.append({"cohort_id": str(cohort_id), "reasons": transition.reasons,
                "state": "unknown", "native_reset_inferred": False})

    def summary(self):
        observed = sum(self.decisions.values())
        return {
            "schema": SCHEMA, "status": "partial_unpriced" if observed else "unobserved",
            "qualification": dict(self.qualification), "successful_cohort_count": self.successful_cohorts,
            "observed_cohort_count": self.observed_cohorts, "unobserved_cohort_count": sum(self.unobserved_reasons.values()),
            "failed_cohort_count": self.failed_cohorts, "observed_ubatch_transition_count": observed,
            "decision_counts": dict(self.decisions), "reason_counts": dict(self.reasons),
            "unobserved_reasons": dict(self.unobserved_reasons), "transitions": tuple(self.records),
            "failures": tuple(self.failure_records), "retained_transition_count": len(self.records),
            "dropped_transition_count": self.dropped_records, "all_observed_records_retained": self.dropped_records == 0,
            "service_cost_ns": None, "duration_adjustment_ns": 0.0, "cost_parameters_added": 0,
            "timing_qualification": "unchanged_unpriced_structure_only", "native_reuse_verified": False,
            "state_previous_unknown": self.state.previous_unknown,
        }
