"""Explicit RTX Blackwell source/PTX integer-warp issue lower bound.

The four 32-thread/clock dispatch partitions in NVIDIA's v1.1 Figure 5
bound ordinary integer warp issue, not actual DP4A execution throughput.
Native compiler mapping and the declared clock are explicit conditions.
No scalar-lane rate, occupancy factor or TensorCore efficiency is borrowed.
Source: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
PTX: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from collections.abc import Mapping
from .mmvq_work import MMVQSourceContract, MMVQWork, SOURCE_SHA256, UnsupportedMMVQ, derive_mmvq_work

SCHEMA = "heterollm.mmvq-integer-warp-issue-bound/v1"
HARDWARE_DOCUMENT_SHA256 = "906ff2a409d7a7e4cbc56f5d3a179d574120d19aaba99520670e1a0c064595fa"
SOURCE_RULE = "rtx-blackwell-v1.1-figure5-four-32-thread-dispatch-partitions"
INSTRUCTION_MODEL = "source_ptx_weight_dependent_dp4a_integer_warp_issue"
CLOCK_CONDITION = "declared_sm_clock_not_wall_time_frequency_upper_bound"

@dataclass(frozen=True)
class MMVQIssueContract:
    runtime_binary_sha256: str
    sm_count: int
    compute_capability: int
    hardware_document_sha256: str
    source_rule: str
    instruction_model: str
    clock_condition: str
    dispatch_partitions_per_sm: int
    threads_per_partition_cycle: int
    warp_size: int

    def __post_init__(self):
        if (not isinstance(self.runtime_binary_sha256, str) or len(self.runtime_binary_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.runtime_binary_sha256)):
            raise UnsupportedMMVQ("issue-bound runtime binary identity required")
        if type(self.sm_count) is not int or self.sm_count <= 0:
            raise UnsupportedMMVQ("issue-bound SM count must be a positive integer")
        expected = {"compute_capability": 1200, "hardware_document_sha256": HARDWARE_DOCUMENT_SHA256,
            "source_rule": SOURCE_RULE, "instruction_model": INSTRUCTION_MODEL,
            "clock_condition": CLOCK_CONDITION, "dispatch_partitions_per_sm": 4,
            "threads_per_partition_cycle": 32, "warp_size": 32}
        for key, value in expected.items():
            actual = getattr(self, key)
            if type(actual) is not type(value) or actual != value:
                raise UnsupportedMMVQ("unverified integer issue upper-bound rule: " + key)

    @classmethod
    def from_mapping(cls, raw: Mapping):
        if not isinstance(raw, Mapping) or raw.get("schema") != SCHEMA:
            raise UnsupportedMMVQ("missing source-bound integer issue contract")
        try:
            return cls(**{key: value for key, value in raw.items() if key != "schema"})
        except TypeError as error:
            raise UnsupportedMMVQ("malformed integer issue contract") from error

    def to_metadata(self):
        return {"schema": SCHEMA, **asdict(self)}


def source_issue_contract(*, runtime_binary_sha256: str, sm_count: int) -> dict:
    """Explicit declaration only: this does not enable the planner switch."""
    return MMVQIssueContract(runtime_binary_sha256, sm_count, 1200, HARDWARE_DOCUMENT_SHA256,
        SOURCE_RULE, INSTRUCTION_MODEL, CLOCK_CONDITION, 4, 32, 32).to_metadata()


def issue_counts(work: MMVQWork) -> dict:
    source = MMVQSourceContract(1200, 1200, 32, dict(SOURCE_SHA256), work.runtime_binary_sha256,
        True, False, True)
    canonical = derive_mmvq_work(m=work.m, k=work.k, n=work.n,
        weight_format=work.weight_format, contract=source, allow_k_formats=True)
    if canonical != work:
        raise UnsupportedMMVQ("MMVQ work differs from immutable source geometry")
    data, correction, chain = {
        "Q5_0": (4, 0, 4), "Q8_0": (2, 0, 2),
        # Q4_K's constant-dot correction can be hoisted across output rows.
        # Retain its source expressions but do not price them as unavoidable.
        "Q4_K": (4, 4, 2), "Q6_K": (2, 0, 1),
    }[work.weight_format]
    steps = tuple(max(work.loop_iterations_by_thread[start:start + 32])
                  for start in range(0, len(work.loop_iterations_by_thread), 32))
    per_warp = tuple(i * work.m * work.rows_per_cta * data for i in steps)
    return {"dp4a_weight_dependent_thread_calls": work.source_vector_dot_calls * data,
        "dp4a_correction_thread_expressions": work.source_vector_dot_calls * correction,
        "dp4a_warp_issue_slots": sum(per_warp) * work.cta_count,
        "dp4a_issue_slots_by_warp_per_cta": per_warp,
        "longest_warp_dp4a_issue_slots": max(per_warp),
        "active_k_warps_per_cta": sum(i > 0 for i in steps),
        "longest_source_dp4a_chain_per_vecdot": chain,
        "longest_source_float_accumulator_updates": work.maximum_thread_k_iterations,
        "independent_accumulators_per_thread": work.m * work.rows_per_cta,
        "dependency_latency_priced": False,
        "partial_warp_accounting": "one_issue_slot_even_when_some_lanes_are_inactive",
        "source_vector_dot_calls": work.source_vector_dot_calls,
        "logical_math_operations": 2 * work.m * work.n * work.k,
        "instruction_count_scope": "source_ptx_conditional_not_verified_SASS"}


def derive_issue_bound(work: MMVQWork, contract: MMVQIssueContract, *, sm_count: int, frequency_ghz: float) -> dict:
    if not isinstance(contract, MMVQIssueContract):
        raise UnsupportedMMVQ("typed integer issue contract required")
    if contract.runtime_binary_sha256 != work.runtime_binary_sha256 or contract.sm_count != sm_count:
        raise UnsupportedMMVQ("integer issue contract runtime/profile binding differs")
    if (isinstance(frequency_ghz, bool) or not isinstance(frequency_ghz, (int, float))
            or not math.isfinite(frequency_ghz) or frequency_ghz <= 0):
        raise UnsupportedMMVQ("positive declared SM clock required")
    counts = issue_counts(work)
    active_sms = min(sm_count, work.cta_count)
    per_partition = contract.threads_per_partition_cycle / contract.warp_size
    capacity = active_sms * contract.dispatch_partitions_per_sm * per_partition
    throughput_cycles = counts["dp4a_warp_issue_slots"] / capacity
    # Serial instruction issue is distinct from (unknown) dependency latency.
    serial_cycles = counts["longest_warp_dp4a_issue_slots"] / per_partition
    cycles = max(throughput_cycles, serial_cycles)
    return {**counts, "m": work.m, "n": work.n, "k": work.k, "weight_format": work.weight_format,
        "cta_count": work.cta_count, "active_sm_upper_bound": active_sms,
        "issue_capacity_warp_slots_per_cycle": capacity,
        "throughput_floor_sm_cycles": throughput_cycles, "serial_warp_issue_floor_cycles": serial_cycles,
        "lower_bound_sm_cycles": cycles, "declared_frequency_ghz": frequency_ghz,
        "service_ns": cycles / frequency_ghz, "capacity_kind": "integer_warp_dispatch_upper_bound",
        "source_contract": contract.to_metadata(), "clock_condition": CLOCK_CONDITION,
        "source_condition": "weight_dependent_source_dp4a_maps_to_native_integer_warp_issue",
        "native_instruction_mapping_proven": False, "dp4a_execution_throughput_known": False,
        "occupancy_efficiency_discount_applied": False,
        "unpriced_work": ("unpack", "constant_dot_correction", "float_scale", "warp_reduction",
                          "barrier_latency", "register_pressure", "spill", "native_instruction_latency"),
        "timing_completeness": "conditional_dot_issue_lower_bound_only"}
