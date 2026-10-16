"""Run paper-parameterized HBM/HBF profiles and a full-backend probe.

The physical profiles deliberately reuse the simulator's DRAM/NAND cores and
remain memory-service diagnostics.  Each nominal profile also runs a small
continuous-batching trace through the V4 GPU/operator/CPU/serving backend.
The trace uses an explicit backend policy: published compute is applied when
available, CIM is disabled for this memory-tier question, and link/host
overrides are reported separately.  It does not reproduce the papers' private
model, topology or workload traces.

Run::

    python scripts/calibrated_hbf_profiles.py --output calibrated_hbf_profiles.json

Units are decimal GB/s, bytes and nanoseconds.  Paper values that are derived
or left unspecified are labelled in the profile metadata.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import sys
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_transfer import summarize_batch
from heterollm_sim.memory_service import DRAM_OPTIONAL_COUNTER_FIELDS
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import report_dict, run_scenario
from heterollm_sim.memory_types import (
    AccessRequest,
    Operation,
)
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.physical_contract import parse_execution_physical_config
from heterollm_sim.cost_models import HostMemoryProfile
from heterollm_sim.ir import (
    ComponentSpec,
    KVCachePolicy,
    LinkSpec,
    PortSpec,
    RequestSpec,
)
from heterollm_sim.nand_core import NandCore


GB = 1_000_000_000
GIB = 1024 ** 3
US = 1_000.0
MS = 1_000_000.0
PAGE = 4096
HBM_REFERENCE_CLOCK_NS = 0.625
HBM_REFERENCE_STACK_BYTES = 64 * GIB


@dataclass(frozen=True)
class CalibratedProfile:
    profile_id: str
    paper: str
    source_url: str
    source_version: str
    hardware: dict[str, Any]
    bandwidth: dict[str, Any]
    latency: dict[str, Any]
    granularity: dict[str, Any]
    buffering: dict[str, Any]
    slo: dict[str, Any]
    workload: dict[str, Any]
    calibration_notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "parameter_provenance": (
                "Selected published parameters checked against the cited arXiv version on 2026-10-13; assumptions and derived service geometry remain explicit."
                if self.profile_id in {"son_hbf", "son_hbf_plus", "hbf_sucks_hbf2_b200", "hot_cold_b200"}
                else "User-supplied or synthetic parameters; no source verification is asserted."
            ),
            "calibrated_to_silicon": False,
            "paper_workload_reproduced": False,
        }


@dataclass(frozen=True)
class BackendExecutionPolicy:
    """Small, explicit overrides for the non-memory backend resources.

    The operator formulas and resource classes stay intact.  This policy only
    selects the aggregate compute scale, host control overhead, interconnect
    envelope and whether the reference CIM path participates in a memory-tier
    experiment.  Missing paper values deliberately keep the bundled reference
    values and are reported as such.
    """

    gpu_compute_pflops: Optional[float] = None
    gpu_compute_source: str = "reference"
    host_control_scale: float = 1.0
    interconnect_bandwidth_gbps: float = 2048.0
    interconnect_latency_ns: float = 100.0
    cim_mode: str = "disabled"

    def __post_init__(self) -> None:
        if self.gpu_compute_pflops is not None and self.gpu_compute_pflops <= 0:
            raise ValueError("gpu_compute_pflops must be positive when supplied")
        if self.host_control_scale <= 0:
            raise ValueError("host_control_scale must be positive")
        if self.interconnect_bandwidth_gbps <= 0:
            raise ValueError("interconnect_bandwidth_gbps must be positive")
        if self.interconnect_latency_ns < 0:
            raise ValueError("interconnect_latency_ns must be non-negative")
        if self.cim_mode not in {"disabled", "reference"}:
            raise ValueError("cim_mode must be disabled or reference")


def _backend_execution_policy(
    profile: CalibratedProfile,
    *,
    gpu_compute_pflops: Optional[float] = None,
    host_control_scale: float = 1.0,
    interconnect_bandwidth_gbps: Optional[float] = None,
    interconnect_latency_ns: Optional[float] = None,
    cim_mode: str = "disabled",
) -> BackendExecutionPolicy:
    paper_compute = profile.hardware.get("compute_pflops")
    target_compute = paper_compute if gpu_compute_pflops is None else gpu_compute_pflops
    source = (
        "paper_aggregate_proxy"
        if paper_compute is not None and gpu_compute_pflops is None
        else "override"
        if gpu_compute_pflops is not None
        else "reference"
    )
    return BackendExecutionPolicy(
        gpu_compute_pflops=(None if target_compute is None else float(target_compute)),
        gpu_compute_source=source,
        host_control_scale=float(host_control_scale),
        interconnect_bandwidth_gbps=float(
            interconnect_bandwidth_gbps
            if interconnect_bandwidth_gbps is not None
            else profile.bandwidth.get(
                "interconnect_bandwidth_gbps",
                float(profile.bandwidth.get("hbf_read_gb_s") or 256.0) * 8.0,
            )
        ),
        interconnect_latency_ns=float(
            interconnect_latency_ns
            if interconnect_latency_ns is not None
            else profile.bandwidth.get("interconnect_latency_ns", 100.0)
        ),
        cim_mode=cim_mode,
    )


def _resource_breakdown(resource_utilization: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Group the full report by hardware owner without hiding raw resources."""

    prefixes = {
        "gpu": ("gpu0.",),
        "cpu": ("cpu0.",),
        "cim": ("cim0.",),
        "interconnect": ("link.", "ucie."),
    }
    result = {}
    for group, group_prefixes in prefixes.items():
        values = {
            key: float(value)
            for key, value in resource_utilization.items()
            if key.startswith(group_prefixes)
        }
        result[group] = {
            "resource_count": len(values),
            "max_utilization": max(values.values(), default=0.0),
            "sum_utilization": sum(values.values()),
            "resources": values,
        }
    return result


def _profile_specs() -> tuple[CalibratedProfile, ...]:
    son_workload = {
        "models": ["Llama 3 405B", "Llama 4 Maverick"],
        "decode_workloads": {
            "short": {"input_tokens": 1660, "output_tokens": 373},
            "mid": {"input_tokens": 5900, "output_tokens": 499},
            "long": {"input_tokens": 103500, "output_tokens": 1100},
        },
        "continuous_batching": True,
        "parallelism": {"intra_node_nvlink_gb_s": 1800, "inter_node_ib_gb_s": 100},
    }
    son_common = dict(
        paper="Exploring High-Bandwidth Flash for Modern LLM Inference",
        source_url="https://arxiv.org/html/2608.13868v1",
        source_version="v1",
        latency={
            "hbf_read_us": 1.0,
            "hbf_write_us": 100.0,
            "source": "paper Section III/Table I",
        },
        granularity={
            "page_bytes": 4096,
            "page_read_latency_us": 1.0,
            "page_program_latency_us": 100.0,
            "hbf_planes_per_core_die": 25,
            "parallelism_mapping": "effective lane count derived from aggregate bandwidth * page latency / page size; not a disclosed channel geometry",
            "source": "paper Section III/Table I",
        },
        buffering={
            "per_hbf_stack_sram_bytes": int(3.13 * 1_000_000),
            "double_buffer": True,
            "write_overlap": True,
            "source": "paper Section III; represented as metadata, not a full overlap model",
        },
        slo={
            "metric": "TPOT",
            "threshold_s": 0.1,
            "sensitivity_s": [0.05, 0.1, 0.2],
            "offline_limit_s": 24 * 3600,
        },
        workload=son_workload,
        calibration_notes=(
            "The paper does not disclose GPU FLOPS/clock; this profile only runs the memory path.",
            "The table's aggregate HBF bandwidth is mapped to one effective host tier; the internal leg is non-bottleneck.",
            "HBF write bandwidth is a paper configuration assumption, not a silicon measurement.",
        ),
    )
    profiles = [
        CalibratedProfile(
            profile_id="son_hbf",
            hardware={
                "gpu_count": 1,
                "hbm_stacks": 1,
                "hbf_stacks": 7,
                "hbm_capacity_gb": 36,
                "hbf_capacity_gb": 7 * 512,
                "total_capacity_gb": 3620,
                "hbf_sram_bytes": 7 * int(3.13 * 1_000_000),
                "source": "Table I, single-GPU configuration",
            },
            bandwidth={
                "hbm_read_gb_s": 1600,
                "hbm_write_gb_s": 1600,
                "hbm_interface_gb_s": 1600,
                "hbm_interface_source": "explicit aggregate interface assumption for this paper probe; not a disclosed pin speed",
                "hbf_read_gb_s": 11200,
                "hbf_write_gb_s": 112,
                "source": "Table I; HBF write is aggregate modeled bandwidth",
            },
            **son_common,
        ),
        CalibratedProfile(
            profile_id="son_hbf_plus",
            hardware={
                "gpu_count": 1,
                "hbm_stacks": 0,
                "hbf_stacks": 8,
                "hbm_capacity_gb": 0,
                "hbf_capacity_gb": 8 * 512,
                "total_capacity_gb": 4096,
                "logic_sram_bytes_per_stack": 40 * 1_000_000,
                "logic_sram_bytes": 8 * 40 * 1_000_000,
                "hbf_sram_bytes": 8 * int(3.13 * 1_000_000),
                "source": "Table I, single-GPU configuration",
            },
            bandwidth={
                "hbm_read_gb_s": 0,
                "hbm_write_gb_s": 0,
                "hbf_read_gb_s": 12800,
                "hbf_write_gb_s": 128,
                "source": "Table I; HBF+ aggregate modeled bandwidth",
            },
            **son_common,
        ),
        CalibratedProfile(
            profile_id="hbf_sucks_hbf2_b200",
            paper="A Full-Stack Characterization of High-Bandwidth Flash for KV-Centric LLM Serving",
            source_url="https://arxiv.org/html/2608.11668v4",
            source_version="v4",
            hardware={
                "gpu_count": 8,
                "local_tier": "B200 HBM3e",
                "near_tier_capacity_gb_per_gpu": 96,
                "near_tier_bandwidth_gb_s_per_gpu": 4000,
                "secondary_tier": "HBF-2",
                "secondary_capacity_tb": 24,
                "secondary_bandwidth_gb_s": 1200,
                "parallelism": "TP8, PP1, DP1",
                "expert_parallelism": "enabled for MoE models",
                "effective_parallel_units": 128,
                "effective_parallel_units_source": "modeled sensitivity anchor; physical HBF channel/plane count is not disclosed",
                "source": "Table 3(c); secondary bandwidth is one modeled tier value",
            },
            bandwidth={
                "hbm_read_gb_s": 4000,
                "hbm_write_gb_s": 4000,
                "hbm_interface_gb_s": 4000,
                "hbm_interface_source": "explicit aggregate interface assumption for this paper probe; not a disclosed pin speed",
                "hbf_read_gb_s": 1200,
                "hbf_write_gb_s": 1200,
            },
            latency={
                "hbf_read_us": 20,
                "hbf_write_us": 200,
                "sweep_us_read_write": [[8, 80], [12, 120], [20, 200], [30, 300]],
                "source": "Section 4.1/6.1; selected HBF-2 mid point",
            },
            granularity={
                "read_bytes": 64,
                "write_page_bytes": 4096,
                "kv_block_tokens": [16, 128, 512],
                "source": "OCP rules and Section 4.1",
            },
            buffering={
                "controller_page_cache": "two page-cache buffers per bank (OCP §5.3)",
                "source": "OCP/Section 8.1",
            },
            slo={
                "metric": "maximum SLO goodput",
                "threshold_s": None,
                "offered_load_qps": [0.05, 32.0],
                "source": "paper reports a max-goodput comparison; absolute SLO threshold is not exposed in Table 3",
            },
            workload={
                "models": ["Qwen3-4B", "Qwen3-32B", "DeepSeek-V3.2-685B", "GLM-5.2-753B", "Kimi-K2.7-Code-1.1T"],
                "trace_count": 4,
                "trace_duration_h": 2,
                "offered_load_qps": [0.05, 32.0],
                "kv_block_tokens": 16,
            },
            calibration_notes=(
                "Table 3(c) bandwidth is a shared secondary-tier modeled value; it is mapped to the host leg and not counted twice.",
                "The paper supplies full TokenSim trace scheduling, near-tier capacity changes and queueing; this profile only supplies the physical memory path.",
            ),
        ),
        CalibratedProfile(
            profile_id="hot_cold_b200",
            paper="Hot–Cold Tiering of HBM and High Bandwidth Flash for Agentic LLM Serving",
            source_url="https://arxiv.org/html/2609.25782v1",
            source_version="v1",
            hardware={
                "gpu_count": 1,
                "gpu": "B200",
                "compute_pflops": 2.25,
                "hbm_capacity_gib": 192,
                "hbf_capacity_tib": 3,
                "hbf_pe_cycles": 100000,
                "effective_parallel_units": 128,
                "effective_parallel_units_source": "modeled sensitivity anchor; physical channel/plane count is not disclosed",
                "source": "Section IV",
            },
            bandwidth={
                "hbm_read_gb_s": 8000,
                "hbm_write_gb_s": 8000,
                "hbm_interface_gb_s": 8000,
                "hbm_interface_source": "explicit aggregate interface assumption for this paper probe; not a disclosed pin speed",
                "hbf_read_gb_s": 8000,
                "hbf_write_gb_s": None,
                "hbf_bandwidth_basis": "matching HBM; write bandwidth not separately disclosed",
            },
            latency={
                "hbf_read_us": 25,
                "resume_overhead_us_paper": 84,
                "source": "Section IV; resume overhead at N_a=48",
            },
            granularity={
                "kv_block_bytes": int(1.5 * 1024 * 1024),
                "kv_block_tokens": "approximately 16-token hash block is not used; 1.5 MiB is the paper's migration block",
                "physical_page_bytes": None,
                "source": "Section IV; physical NAND page geometry is not disclosed",
            },
            buffering={
                "base_die_double_buffer": True,
                "write_policy": "write-on-evict",
                "prefetch": "one step ahead on resume/decode",
                "waf": 1.02,
                "source": "Section III/IV",
            },
            slo={
                "metric": "time-between-tokens",
                "tbt_s": 0.014,
                "resume_s": 0.000084,
                "slo_tbt_s": 0.050,
                "active_sessions": [8, 48, 128],
            },
            workload={
                "model": "Qwen3-Coder-30B-A3B",
                "weights_gib": 56.8,
                "kv_bytes_per_token": 98304,
                "peak_context_tokens": 16700,
                "resident_agent_gib": 0.82,
                "idle_median_s": 8,
                "idle_mean_s": 23,
            },
            calibration_notes=(
                "The 1.5 MiB block is modeled as a service granularity, not as a physical NAND page.",
                "The paper does not disclose HBF write bandwidth, physical page/channel geometry or buffer capacity; the read-only resume run does not charge HBF writes.",
                "Original trajectories and in-house simulator are not published, so 165-to-3900 sessions cannot be replayed here.",
            ),
        ),
    ]
    return tuple(profiles)


def _profile_by_id(profile_id: str) -> CalibratedProfile:
    for profile in _profile_specs():
        if profile.profile_id == profile_id:
            return profile
    raise KeyError(profile_id)


def _hbm_physical_config(*, capacity_bytes: int, read_gb_s: float, write_gb_s: float,
                         stacks: Optional[int] = None,
                         interface_gb_s: Optional[float] = None,
                         clock_ns: Optional[float] = None,
                         interface_input_scope: Optional[str] = None,
                         parameter_profile: Optional[str] = None,
                         profile_id: str = "hbm3e_reference"):
    """Apply a paper/user capacity budget to an explicitly selected reference.

    The paper's aggregate capacity and rates do not identify a manufacturer's
    die density, stack height or timings. Preserve their declared units and
    let the shared catalog select and label a covering reference organization.
    """
    if interface_gb_s is not None and clock_ns is not None:
        raise ValueError("choose an independent HBM clock or physical interface bandwidth, not both")
    selected_stacks = (stacks if stacks is not None else
                       max(1, (capacity_bytes + HBM_REFERENCE_STACK_BYTES - 1) // HBM_REFERENCE_STACK_BYTES))
    overrides = {"capacity_bytes": capacity_bytes,
        "stacks": selected_stacks, "dies_per_stack": 16, "die_density_gbits": 32,
        "interface_bandwidth_gb_s": interface_gb_s,
        "read_bandwidth_gb_s": read_gb_s, "write_bandwidth_gb_s": write_gb_s,
        "metadata": {
            "model_abstraction": "analytical_capacity_bandwidth_comparison",
            "target_hardware_calibrated": False,
            "stack_organization_assumption": "fixed 16H x 32 Gbit, 64 GiB per stack; capacity changes select stack count, not density or frequency",
            "capacity_input_scope": "paper_or_user_usable_capacity_budget",
            "capacity_input_units": "GB inputs use 10^9 bytes; GiB inputs use 2^30 bytes; neither identifies a physical DRAM part",
            "source_stack_count": stacks,
            "source_stack_count_status": "declared_input" if stacks is not None else "unknown_reference_selected",
            "source_device_geometry_known": False,
            "interface_input_scope": interface_input_scope or (
                "independent_reference_command_clock"
                if interface_gb_s is None else "declared_physical_interface_budget"),
        }}
    if interface_gb_s is None:
        overrides.update(bandwidth_input_mode="data_rate",
                         command_clock_ns=HBM_REFERENCE_CLOCK_NS if clock_ns is None else clock_ns)
        overrides["metadata"]["clock_assumption"] = (
            "0.625 ns default is an explicit analytical reference, not a vendor speed-bin guarantee; directional budgets do not set the clock"
            if clock_ns is None else "explicit independent command clock; not a vendor calibration")
    config = resolve_physical_config(profile_id, overrides=overrides)
    # The normal HBM3 service reference includes all ordinary-access gates.
    # Other authored geometry/timing controls keep their separate research
    # meaning unless the caller explicitly selects a parameter pack.
    selected_parameters = (parameter_profile if parameter_profile is not None else
        "reference" if profile_id == "hbm3e_reference" else "none")
    from heterollm_sim.hbm_timing_profiles import apply_hbm_timing_profile
    return apply_hbm_timing_profile(config, selected_parameters)


def _hbm_core(*, capacity_bytes: int, read_gb_s: float, write_gb_s: float,
              stacks: Optional[int] = None,
              interface_gb_s: Optional[float] = None,
              clock_ns: Optional[float] = None,
              parameter_profile: Optional[str] = None,
              profile_id: str = "hbm3e_reference") -> DramCore:
    return DramCore(_hbm_physical_config(capacity_bytes=capacity_bytes,
        read_gb_s=read_gb_s, write_gb_s=write_gb_s, stacks=stacks,
        interface_gb_s=interface_gb_s, clock_ns=clock_ns, profile_id=profile_id,
        parameter_profile=parameter_profile), capture_details=False)


def _hbf_core(*, capacity_bytes: int, read_gb_s: float, write_gb_s: float,
              page_bytes: Optional[int] = None, read_latency_ns: Optional[float] = None,
              program_ns: Optional[float] = None, parallel_units: Optional[int] = None,
              host_granularity_bytes: Optional[int] = None,
              host_channels: int = 1, host_full_duplex: bool = False,
              read_cache_pages_per_bank: int = 0) -> NandCore:
    """Resolve unspecified geometry and timing in the shared catalog."""
    overrides = {"capacity_bytes": capacity_bytes,
        "host_bandwidth_gb_s": max(read_gb_s, write_gb_s),
        "host_read_bandwidth_gb_s": read_gb_s,
        "host_write_bandwidth_gb_s": write_gb_s,
        "write_buffer_bytes": 0, "protocol_mode": "generic_nand",
        "host_channels": host_channels, "host_full_duplex": host_full_duplex,
        "read_cache_pages_per_bank": read_cache_pages_per_bank,
        "metadata": {"model_abstraction": "analytical_page_service",
            "model_scope": "logical model buffers with RMW; no OCP physical write-order claim or invented FTL cost",
            "target_hardware_calibrated": False,
            "address_scope": "logical model buffers mapped directly to equivalent page services; no FTL or physical OCP write-order claim",
            "parallelism_basis": "explicit effective service units or the catalog's fixed sensitivity assumption; never inferred from desired interface throughput"}}
    overrides.update({key: value for key, value in {
        "page_bytes": page_bytes, "page_read_ns": read_latency_ns,
        "page_program_ns": program_ns, "parallel_units": parallel_units,
        "host_granularity_bytes": host_granularity_bytes}.items() if value is not None})
    config = resolve_physical_config("published_hbf", overrides=overrides,
        expand_geometry=True)
    return NandCore(config, capture_details=False)


def hbf_service_summary(config) -> dict[str, Any]:
    """Separate interface budgets from ideal cold-array service ceilings."""
    units = config.array_units
    def ceiling(latency):
        return units * config.page_bytes / latency if latency > 0 else None
    return {
        "model_abstraction": "analytical_page_service" if config.protocol_mode == "generic_nand" else "ocp_physical_protocol_subset",
        "protocol_mode": config.protocol_mode,
        "host_channels": config.host_channels,
        "host_full_duplex": config.host_full_duplex,
        "read_cache_pages_per_bank": config.read_cache_pages_per_bank,
        "interface_read_cap_gb_s": config.host_bandwidth_for("read"),
        "interface_write_cap_gb_s": config.host_bandwidth_for("write"),
        "ideal_cold_array_read_ceiling_gb_s": ceiling(config.page_read_ns),
        "ideal_array_program_ceiling_gb_s": ceiling(config.page_program_ns),
        "effective_parallel_units": units,
        "page_bytes": config.page_bytes, "host_granularity_bytes": config.host_granularity_bytes,
        "page_read_ns": config.page_read_ns, "page_program_ns": config.page_program_ns,
        "ceiling_scope": "units * page bytes / array latency; excludes transfer, commands, erase, RMW and queueing; read cache hits do not use this cold-array ceiling",
        "calibrated_to_silicon": False,
    }


def hbm_protocol_summary(config) -> dict[str, Any]:
    """Expose recorded protocol scope without filling undisclosed values."""
    values = config if isinstance(config, dict) else asdict(config)
    metadata = values.get("metadata", {})
    return {
        **{key: values.get(key) for key in (
            "generation", "control_mode", "command_clock_ns", "maximum_active_ns",
            "column_same_group_ns", "column_different_group_ns", "precharge_to_precharge_ns",
            "minimum_active_ns", "precharge_to_activate_ns", "activate_cycle_ns",
            "read_latency_ns", "write_latency_ns", "refresh_policy",
            "refresh_interval_ns", "refresh_cycle_ns", "bandwidth_input_mode",
            "interface_bandwidth_gb_s", "data_rate_mt_s", "effective_pin_data_rate_gbps",
            "read_bandwidth_gb_s", "write_bandwidth_gb_s",
            "address_mapping", "scheduler", "scheduler_window", "queue_scope", "max_outstanding_requests",
            "activate_same_group_ns", "activate_different_group_ns", "four_activate_window_ns",
            "read_to_precharge_ns", "write_to_read_same_group_ns", "write_to_read_different_group_ns",
            "read_to_write_command_ns")},
        **{key: metadata.get(key) for key in (
            "geometry_status", "control_status", "timing_validation", "timing_constraint_validation", "timing_basis",
            "unknown_parameters", "unconfigured_timing_parameters", "control_limitations", "accounting_basis",
            "interface_input_scope", "operating_point_basis", "directional_cap_basis", "hbm_parameter_profile")},
    }


def _request(core: Any, *, operation: Operation, address: int, bytes_: int,
             arrival_ns: float, label: str):
    return core.submit(AccessRequest(label, operation, address, bytes_, arrival_ns))


def _stream(core_hbm: Optional[DramCore], core_hbf: Optional[NandCore], *,
            kv_pages: int, steps: int, kv_read_tier: str,
            kv_write_tier: str, weight_tier: str,
            traffic_page_bytes: int = PAGE) -> dict[str, Any]:
    if kv_pages < 2 or steps < 1:
        raise ValueError("kv_pages must be at least 2 and steps must be positive")
    cores = {"hbm": core_hbm, "hbf": core_hbf}
    if any(cores[tier] is None for tier in {kv_read_tier, kv_write_tier, weight_tier}):
        raise ValueError("requested tier is not present in this profile")
    by_tier: dict[str, list[Any]] = {"hbm": [], "hbf": []}
    clock = 0.0
    step_service_ns = []
    for step in range(steps):
        current = []
        weight = _request(
            cores[weight_tier], operation=Operation.READ,
            address=max(16, kv_pages) * traffic_page_bytes,
            bytes_=8 * traffic_page_bytes, arrival_ns=clock, label=f"weight-{step}",
        )
        by_tier[weight_tier].append(weight)
        current.append(weight)
        for page in range(kv_pages):
            item = _request(
                cores[kv_read_tier], operation=Operation.READ,
                address=page * traffic_page_bytes,
                bytes_=traffic_page_bytes,
                arrival_ns=clock, label=f"kv-read-{step}-{page}",
            )
            by_tier[kv_read_tier].append(item)
            current.append(item)
        for page in (kv_pages - 2, kv_pages - 1):
            item = _request(
                cores[kv_write_tier], operation=Operation.WRITE,
                address=page * traffic_page_bytes,
                bytes_=traffic_page_bytes,
                arrival_ns=clock, label=f"kv-write-{step}-{page}",
            )
            by_tier[kv_write_tier].append(item)
            current.append(item)
        completion = max(item.completion_ns for item in current)
        step_service_ns.append(completion - clock)
        clock = completion
    logical_bytes = sum(item.logical_bytes for values in by_tier.values() for item in values)
    tiers: dict[str, Any] = {}
    for tier, values in by_tier.items():
        if not values:
            tiers[tier] = {"logical_bytes": 0, "physical_read_bytes": 0, "physical_write_bytes": 0}
            continue
        batch = summarize_batch(values)
        tiers[tier] = {
            "logical_bytes": sum(item.logical_bytes for item in values),
            "logical_read_bytes": sum(item.logical_bytes for item in values if item.operation is Operation.READ),
            "logical_write_bytes": sum(item.logical_bytes for item in values if item.operation is Operation.WRITE),
            "physical_read_bytes": batch.physical_read_bytes,
            "physical_write_bytes": batch.physical_write_bytes,
            "queue_wait_ns": batch.queue_wait_ns,
            "write_buffer_wait_ns": batch.write_buffer_wait_ns,
            **({"protocol": hbm_protocol_summary(cores[tier].config),
                **{key: batch.counters.get(key) for key in DRAM_OPTIONAL_COUNTER_FIELDS}}
               if tier == "hbm" else {}),
        }
    return {
        "service_ns": float(clock),
        "service_us": float(clock / US),
        "step_service_ns": step_service_ns,
        "max_step_service_ns": max(step_service_ns),
        "logical_bytes": int(logical_bytes),
        "useful_bandwidth_gb_s": float(logical_bytes / clock) if clock else None,
        "tiers": tiers,
        "traffic": {"kv_pages": kv_pages, "steps": steps, "page_bytes": traffic_page_bytes},
    }


def _profile_hardware(profile: CalibratedProfile, *,
                      latency_override_us: Optional[tuple[float, float]] = None) -> tuple[Optional[DramCore], Optional[NandCore], dict[str, Any]]:
    p = profile
    bw = p.bandwidth
    hbf_read = float(bw.get("hbf_read_gb_s", 0) or 0)
    hbf_write_declared = bw.get("hbf_write_gb_s")
    hbf_write = float(hbf_write_declared or 0)
    applied = "nominal"
    hbm_read = float(bw.get("hbm_read_gb_s", 0) or 0)
    hbm_write = float(bw.get("hbm_write_gb_s", 0) or 0)
    hbm_cap = p.hardware.get("hbm_capacity_gb")
    if hbm_cap is None:
        hbm_cap = p.hardware.get("near_tier_capacity_gb_per_gpu")
    if hbm_cap is None:
        hbm_cap = p.hardware.get("hbm_capacity_gib")
        if hbm_cap is not None:
            hbm_cap = float(hbm_cap) * GIB / GB
    hbm = None
    if hbm_read > 0 and hbm_cap and hbm_cap > 0:
        hbm = _hbm_core(
            capacity_bytes=max(PAGE * 64, int(float(hbm_cap) * GB)),
            read_gb_s=hbm_read,
            write_gb_s=hbm_write or hbm_read,
            interface_gb_s=bw.get("hbm_interface_gb_s"),
            clock_ns=bw.get("hbm_command_clock_ns"),
            stacks=int(p.hardware["hbm_stacks"]) if p.hardware.get("hbm_stacks") else None,
        )
    hbf_cap = p.hardware.get("hbf_capacity_gb")
    if hbf_cap is None and p.hardware.get("secondary_capacity_tb"):
        hbf_cap = float(p.hardware["secondary_capacity_tb"]) * 1000.0
    if hbf_cap is None and p.hardware.get("hbf_capacity_tib"):
        hbf_cap = float(p.hardware["hbf_capacity_tib"]) * (1024 ** 4) / GB
    hbf = None
    if hbf_read > 0 and hbf_cap and hbf_cap > 0:
        if hbf_write <= 0:
            # A read-only profile still needs a valid constructor field.  It is
            # never used by the Hot-Cold resume experiment and is reported as
            # unspecified rather than treated as a measured write bandwidth.
            hbf_write = hbf_read
        granularity = p.granularity
        page_bytes = (granularity.get("page_bytes") or granularity.get("write_page_bytes")
                      or granularity.get("kv_block_bytes"))
        read_us = p.latency.get("hbf_read_us", granularity.get("page_read_latency_us"))
        program_us = p.latency.get("hbf_write_us", granularity.get("page_program_latency_us"))
        if latency_override_us is not None:
            read_us, program_us = latency_override_us
        read_ns = float(read_us) * US if read_us is not None else None
        program_ns = float(program_us) * US if program_us is not None else None
        parallel_units = p.hardware.get("effective_parallel_units")
        host_granularity = int(p.granularity.get("read_bytes", 0) or 0)
        if host_granularity <= 0 or (page_bytes is not None and host_granularity > page_bytes):
            host_granularity = None
        hbf = _hbf_core(
            capacity_bytes=int(float(hbf_cap) * GB),
            read_gb_s=hbf_read,
            write_gb_s=hbf_write or hbf_read,
            page_bytes=page_bytes,
            read_latency_ns=read_ns,
            program_ns=program_ns,
            parallel_units=parallel_units,
            host_granularity_bytes=host_granularity,
            host_channels=p.hardware.get("hbf_host_channels", 1),
            host_full_duplex=p.hardware.get("hbf_host_full_duplex", False),
            read_cache_pages_per_bank=p.buffering.get("read_cache_pages_per_bank", 0),
        )
    return hbm, hbf, {
        "bandwidth_variant": applied,
        "hbm_read_gb_s": hbm_read,
        "hbm_write_gb_s": hbm_write,
        "hbm_physical_config": asdict(hbm.config) if hbm else None,
        "hbm_protocol": hbm_protocol_summary(hbm.config) if hbm else None,
        "hbf_read_gb_s": hbf_read,
        "hbf_write_gb_s": hbf_write,
        "hbf_write_was_unspecified": hbf_write_declared is None,
        "hbf_page_bytes": hbf.config.page_bytes if hbf else None,
        "hbf_parallel_units": hbf.config.parallel_units if hbf else None,
        "hbf_host_granularity_bytes": hbf.config.host_granularity_bytes if hbf else None,
        "hbf_read_latency_ns": hbf.config.page_read_ns if hbf else None,
        "hbf_program_ns": hbf.config.page_program_ns if hbf else None,
        "hbf_service": hbf_service_summary(hbf.config) if hbf else None,
    }


def _full_backend_scenario(
    profile: CalibratedProfile,
    *,
    use_hbf: bool = True,
    request_count: int = 2,
    prompt_tokens: int = 128,
    output_tokens: int = 16,
    gpu_compute_pflops: Optional[float] = None,
    host_control_scale: float = 1.0,
    interconnect_bandwidth_gbps: Optional[float] = None,
    interconnect_latency_ns: Optional[float] = None,
    cim_mode: str = "disabled",
    hbm_timing_profile: Optional[str] = None,
):
    """Build a small continuous-batching scenario on the real V4 backend.

    The physical profile remains a memory microbenchmark.  This companion
    scenario uses the repository's normal control-plane, GPU operator cost
    model, HBM resource queues, CPU submission path and serving scheduler.  It
    intentionally uses a small trace so the experiment stays repeatable; it
    is not a replay of any paper's private workload trace.
    """

    base = build_reference_scenario()
    backend_policy = _backend_execution_policy(
        profile,
        gpu_compute_pflops=gpu_compute_pflops,
        host_control_scale=host_control_scale,
        interconnect_bandwidth_gbps=interconnect_bandwidth_gbps,
        interconnect_latency_ns=interconnect_latency_ns,
        cim_mode=cim_mode,
    )
    if request_count < 1 or prompt_tokens < 1 or output_tokens < 1:
        raise ValueError("request_count, prompt_tokens and output_tokens must be positive")
    bw = profile.bandwidth
    hbm_read = float(bw.get("hbm_read_gb_s", 0) or 0)
    # HBF+ has no local HBM stack. Its separate comparison/support tier uses
    # paper capacity/rates and a JESD270-4A reference organization, not a known SKU.
    if hbm_read <= 0:
        hbm_read = 12_800.0
    hbm_write = float(bw.get("hbm_write_gb_s", 0) or hbm_read)
    hbm_capacity_gb = profile.hardware.get("hbm_capacity_gb")
    if hbm_capacity_gb is None:
        hbm_capacity_gb = profile.hardware.get("near_tier_capacity_gb_per_gpu")
    if hbm_capacity_gb is None:
        hbm_capacity_gb = profile.hardware.get("hbm_capacity_gib")
        if hbm_capacity_gb is not None:
            hbm_capacity_gb = float(hbm_capacity_gb) * GIB / GB
    if not hbm_capacity_gb or float(hbm_capacity_gb) <= 0.0:
        hbm_capacity_gb = 288.0 if profile.profile_id == "son_hbf_plus" else 16.0
    hbm_capacity = max(
        16 * GIB,
        int(float(hbm_capacity_gb or 16.0) * GB),
    )
    hbm_stacks = (int(profile.hardware["hbm_stacks"]) if profile.hardware.get("hbm_stacks")
                  else 8 if profile.profile_id == "son_hbf_plus" else None)
    physical_hbm = _hbm_physical_config(capacity_bytes=hbm_capacity,
        read_gb_s=hbm_read, write_gb_s=hbm_write, stacks=hbm_stacks,
        interface_gb_s=bw.get("hbm_interface_gb_s"),
        clock_ns=bw.get("hbm_command_clock_ns"),
        profile_id=hbm_timing_profile or (
            "hbm4_reference" if profile.profile_id == "son_hbf_plus" else "hbm3e_reference"))
    hbm0 = base.hardware.get_component("hbm0")
    hbm_link_gbps = physical_hbm.physical_interface_bandwidth_gb_s * 8.0
    hbm_link_version = "4.0" if physical_hbm.generation == "HBM4" else "3.0"
    hbm0 = replace(
        hbm0,
        capacity_bytes=hbm_capacity,
        bandwidth_gbps=hbm_link_gbps,
        read_bandwidth_gbps=hbm_read * 8.0,
        write_bandwidth_gbps=hbm_write * 8.0,
        ports=tuple(
            replace(port, bandwidth_gbps=hbm_link_gbps, version=hbm_link_version)
            for port in hbm0.ports
        ),
        metadata={**hbm0.metadata, "physical_memory_config": physical_hbm},
    )
    gpu0 = base.hardware.get_component("gpu0")
    ucie_bandwidth_gbps = backend_policy.interconnect_bandwidth_gbps
    gpu0 = replace(
        gpu0,
        ports=tuple(
            replace(port, bandwidth_gbps=hbm_link_gbps, version=hbm_link_version)
            if port.port_id == "hbm0" else port
            for port in gpu0.ports
        ) + (PortSpec(
            port_id="ucie_hbf", protocol="UCIe", role="controller",
            version="1.1", lanes=64, bandwidth_gbps=ucie_bandwidth_gbps,
            payload="streaming",
        ),),
    )
    # Share the exact memory service configuration with the physical profile.
    # Previously this path silently used the library's 488 GB/s internal bus,
    # dropped the stack multiplier, and rounded a 64-byte read to 4 KiB.
    _, hbf_core, _ = _profile_hardware(profile)
    if hbf_core is None:
        raise ValueError("full HBF probe requires an HBF profile")
    hbf_physical = hbf_core.config
    hbf_capacity = hbf_physical.effective_capacity_bytes
    hbf_read = float(hbf_physical.host_read_bandwidth_gb_s)
    hbf_write = float(hbf_physical.host_write_bandwidth_gb_s)
    hbf_parallel = int(hbf_physical.parallel_units)
    hbf_read_us = hbf_physical.page_read_ns / US
    hbf_write_us = hbf_physical.page_program_ns / US
    hbf_profile = HostMemoryProfile(
        bandwidth_gb_s=max(hbf_read, hbf_write, 1.0),
        efficiency=1.0,
        read_bandwidth_gb_s=max(hbf_read, 1.0),
        write_bandwidth_gb_s=max(hbf_write, 1.0),
        resource_id="hbf0.memory",
        name=f"{profile.profile_id}-hbf",
        read_latency_ns=hbf_read_us * US,
        write_latency_ns=hbf_write_us * US,
        transaction_bytes=hbf_physical.host_granularity_bytes,
        max_outstanding_requests=32,
        parallel_lanes=hbf_parallel,
    )
    hbf = ComponentSpec(
        component_id="hbf0",
        kind="hbf",
        cost_profile_id="calibrated-hbf",
        ports=(PortSpec(
            port_id="ucie0", protocol="UCIe", role="endpoint", version="1.1",
            lanes=64, bandwidth_gbps=ucie_bandwidth_gbps, payload="streaming",
        ),),
        package_id="package0",
        die_id="hbf_die",
        capacity_bytes=hbf_capacity,
        bandwidth_gbps=max(hbf_read, hbf_write, 1.0) * 8.0,
        read_bandwidth_gbps=max(hbf_read, 1.0) * 8.0,
        write_bandwidth_gbps=max(hbf_write, 1.0) * 8.0,
        metadata={
            "physical_memory_config": hbf_physical,
            # HBF is an active, writable, directly exposed device buffer in
            # this probe.  This is what makes _add_kv_access use
            # endpoint_service/PhysicalService and the NAND core rather than
            # being treated as a DRAM profile.
            "access_mode": "memory",
            "writable": True,
            "write_buffer_bytes": 0,
            "memory_service_owner": hbf_profile.resource_id,
            "read_latency_ns": hbf_profile.read_latency_ns,
            "write_latency_ns": hbf_profile.write_latency_ns,
            "transfer_granularity_bytes": hbf_profile.transaction_bytes,
            "max_outstanding_requests": hbf_profile.max_outstanding_requests,
            "parallel_lanes": hbf_profile.parallel_lanes,
        },
    )
    hardware = replace(
        base.hardware,
        components=tuple(
            gpu0 if component.component_id == "gpu0" else
            hbm0 if component.component_id == "hbm0" else component
            for component in base.hardware.components
        ) + (hbf,),
        links=tuple(
            replace(link, bandwidth_gbps=hbm_link_gbps, version=hbm_link_version)
            if {link.source_component, link.target_component} == {"gpu0", "hbm0"}
            else link
            for link in base.hardware.links
        ) + (LinkSpec(
            link_id="gpu-hbf-ucie", source_component="gpu0", source_port="ucie_hbf",
            target_component="hbf0", target_port="ucie0", protocol="UCIe",
            version="1.1", lanes=64, bandwidth_gbps=ucie_bandwidth_gbps,
            latency_ns=backend_policy.interconnect_latency_ns,
            payload="streaming",
        ),),
        metadata={
            **base.hardware.metadata,
            "execution_calibration": {
                "gpu_compute_pflops": backend_policy.gpu_compute_pflops,
                "gpu_compute_source": backend_policy.gpu_compute_source,
                "host_control_scale": backend_policy.host_control_scale,
                "interconnect_bandwidth_gbps": backend_policy.interconnect_bandwidth_gbps,
                "interconnect_latency_ns": backend_policy.interconnect_latency_ns,
                "cim_mode": backend_policy.cim_mode,
                "hbm_backend": "DramCore",
            },
        },
        require_connected=False,
    )
    component_profiles = {
        kind: dict(registry)
        for kind, registry in base.component_profiles.items()
    }
    component_profiles.setdefault("host_memory", {})["calibrated-hbf"] = hbf_profile
    if backend_policy.cim_mode == "disabled":
        hardware = replace(
            hardware,
            components=tuple(
                component for component in hardware.components
                if component.component_id != "cim0"
            ),
            links=tuple(
                link for link in hardware.links
                if link.source_component != "cim0"
                and link.target_component != "cim0"
            ),
        )
        component_profiles.pop("cim", None)
    hbm_profile = component_profiles["hbm"]["legacy-hbm-0"]
    component_profiles["hbm"]["legacy-hbm-0"] = replace(
        hbm_profile,
        bandwidth_gb_s=physical_hbm.physical_interface_bandwidth_gb_s,
        read_bandwidth_gb_s=hbm_read,
        write_bandwidth_gb_s=hbm_write,
        measured_effective_bandwidth_gb_s=None,
        parallel_lanes=physical_hbm.lane_count,
        transaction_bytes=physical_hbm.burst_bytes,
        max_outstanding_requests=physical_hbm.max_outstanding_requests,
    )
    gpu_profile_id = gpu0.cost_profile_id
    gpu_profile = component_profiles["gpu"].get(gpu_profile_id)
    if gpu_profile is None:
        raise ValueError("reference GPU profile {} is missing".format(gpu_profile_id))
    if backend_policy.gpu_compute_pflops is not None:
        target_tops = backend_policy.gpu_compute_pflops * 1_000.0
        scale = target_tops / max(float(gpu_profile.peak_tops), 1e-12)
        component_profiles["gpu"][gpu_profile_id] = replace(
            gpu_profile,
            tensor_core=replace(
                gpu_profile.tensor_core,
                frequency_ghz=gpu_profile.tensor_core.frequency_ghz * scale,
            ),
            name=f"{profile.profile_id}-aggregate-gpu",
        )
    host = base.host_orchestration_profile
    if backend_policy.host_control_scale != 1.0:
        host = replace(
            host,
            request_parse_ns=host.request_parse_ns * backend_policy.host_control_scale,
            batch_fixed_ns=host.batch_fixed_ns * backend_policy.host_control_scale,
            token_pack_ns=host.token_pack_ns * backend_policy.host_control_scale,
            submission_ns=host.submission_ns * backend_policy.host_control_scale,
            dma_queue_submission_ns=host.dma_queue_submission_ns * backend_policy.host_control_scale,
            dma_latency_ns=host.dma_latency_ns * backend_policy.host_control_scale,
            kv_page_lookup_ns=host.kv_page_lookup_ns * backend_policy.host_control_scale,
            kv_descriptor_ns=host.kv_descriptor_ns * backend_policy.host_control_scale,
        )
    parallel = base.placement.parallel
    if backend_policy.cim_mode == "disabled":
        parallel = replace(
            parallel,
            rank_mapping=tuple(
                replace(rank, cim_component_id=None)
                for rank in parallel.rank_mapping
            ),
        )
    requests = tuple(
        RequestSpec(
            request_id=f"{profile.profile_id}-full-{index}",
            arrival_ns=float(index) * 100_000.0,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            priority=0,
        )
        for index in range(max(1, int(request_count)))
    )
    scheduler = replace(
        base.workload.scheduler,
        mode="continuous",
        max_num_seqs=max(1, int(request_count)),
        max_num_batched_tokens=max(128, prompt_tokens),
        max_num_ubatch_tokens=max(128, prompt_tokens),
        prefill_chunk_tokens=max(1, prompt_tokens),
        # The direct-cache probe is a capacity-sufficient single-pass trace;
        # disabling scheduler swaps keeps the baseline and HBF runs identical
        # and avoids introducing an unmeasured second offload tier.
        preemption_enabled=False,
        preemption_policy="auto",
        slo_ttft_ns=None,
        slo_tbt_ns=(
            float(profile.slo.get("slo_tbt_s")) * 1e9
            if profile.slo.get("slo_tbt_s") is not None
            else float(profile.slo.get("threshold_s")) * 1e9
            if profile.slo.get("threshold_s") is not None
            else None
        ),
    )
    workload = replace(
        base.workload,
        requests=requests,
        request_count=len(requests),
        scheduler=scheduler,
        mtp=None,
    )
    # The direct HBF case keeps rank-local HBM for ordinary GPU traffic but
    # exposes HBF as the writable KV endpoint.  _add_kv_access then prices the
    # NAND page service and link instead of treating HBF as a DRAM profile.
    # The paired baseline keeps the same GPU/model/request trace and uses HBM
    # KV, so the difference is attributable to the memory placement.
    cache_component = "hbf0" if use_hbf else "hbm0"
    placement_metadata = dict(base.placement.metadata)
    placement_metadata["llama_backend_memory"] = (
        {
            "hbf0": {
                "device_id": "gpu0",
                "buffer_kind": "HBF",
                "access": "direct",
                "priority": 1,
            }
        }
        if use_hbf
        else {}
    )
    policy = KVCachePolicy(
        cache_component=cache_component,
        offload_component=None,
        tokens_per_page=16,
        dtype="int8",
        layout_mode="legacy_single",
        prefetch_distance=0,
        preemption_mode="auto",
    )
    return replace(
        base,
        hardware=hardware,
        component_profiles=component_profiles,
        placement=replace(
            base.placement,
            hardware_name=hardware.name,
            kv_policy=policy,
            parallel=parallel,
            metadata=placement_metadata,
        ),
        workload=workload,
        host_orchestration_profile=host,
    )


def _full_backend_summary(
    profile: CalibratedProfile,
    *,
    use_hbf: bool = True,
    request_count: int = 2,
    prompt_tokens: int = 128,
    output_tokens: int = 16,
    gpu_compute_pflops: Optional[float] = None,
    host_control_scale: float = 1.0,
    interconnect_bandwidth_gbps: Optional[float] = None,
    interconnect_latency_ns: Optional[float] = None,
    cim_mode: str = "disabled",
    hbm_timing_profile: Optional[str] = None,
) -> dict[str, Any]:
    """Run the calibrated profile through the full GPU/serving backend."""

    backend_policy = _backend_execution_policy(
        profile,
        gpu_compute_pflops=gpu_compute_pflops,
        host_control_scale=host_control_scale,
        interconnect_bandwidth_gbps=interconnect_bandwidth_gbps,
        interconnect_latency_ns=interconnect_latency_ns,
        cim_mode=cim_mode,
    )

    scenario = _full_backend_scenario(
        profile,
        use_hbf=use_hbf,
        request_count=request_count,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        gpu_compute_pflops=gpu_compute_pflops,
        host_control_scale=host_control_scale,
        interconnect_bandwidth_gbps=interconnect_bandwidth_gbps,
        interconnect_latency_ns=interconnect_latency_ns,
        cim_mode=cim_mode,
        hbm_timing_profile=hbm_timing_profile,
    )
    result = run_scenario(scenario)
    report = report_dict(result)
    summary = report["summary"]
    gpu = scenario.resolve_component_profile("gpu0")
    kv = report.get("kv_cache", {})
    storage = report.get("storage_traffic", {})
    dram = report.get("dram_traffic", {})
    throughput = summary.get("throughput", {})
    goodput = summary.get("goodput", {})
    resource_utilization = report.get("resource_utilization", {})
    gpu_resource_busy = {}
    gpu_busy_recorded_batches = 0
    for batch in result.serving.batches:
        metadata = batch.cost.metadata
        if metadata.get("physical_execution_scope") != "persistent_live_kernel":
            continue
        gpu_busy_recorded_batches += 1
        for key, value in metadata.get("resource_busy_ns", {}).items():
            if str(key).startswith("gpu0."):
                gpu_resource_busy[str(key)] = gpu_resource_busy.get(str(key), 0.0) + float(value)
    slo_configured = any(value is not None for value in (
        scenario.workload.scheduler.slo_ttft_ns,
        scenario.workload.scheduler.slo_tbt_ns,
    ))
    return {
        "full_backend_used": True,
        "execution_mode": report.get("execution_mode"),
        "retention_policy": report.get("retention_policy"),
        "backend_model": "TopologyAwareBatchCostProvider + simulate_online",
        "gpu_model": {
            "name": gpu.name,
            "sm_count": gpu.sm_count,
            "peak_tops": gpu.peak_tops,
            "attainable_tops": gpu.attainable_tops,
            "source": "bundled reference GPU profile",
            "calibration_source": backend_policy.gpu_compute_source,
        },
        "paper_compute_pflops": profile.hardware.get("compute_pflops"),
        "paper_compute_applied": backend_policy.gpu_compute_source == "paper_aggregate_proxy",
        "backend_policy": asdict(backend_policy),
        "hbm_control_label": (
            "son_hbm4_reference_not_same_profile"
            if profile.profile_id == "son_hbf_plus"
            else "profile_near_hbm"
        ),
        "memory_placement": "hbf_direct_kv" if use_hbf else "hbm_local_kv",
        "experiment_scope": "synthetic single-GPU KV placement probe; weights and intermediate data use the local HBM support tier",
        "paper_topology_matched": False,
        "assumptions": {
            "interconnect": (
                "explicit override" if interconnect_bandwidth_gbps is not None
                else "endpoint-matched bandwidth proxy; not a measured package link"
            ),
            "hbm_support_tier": (
                "288 decimal GB paper capacity budget over a JESD270-4A HBM4 reference organization; actual vendor SKU and timings are unknown, and this support tier is absent from the paper HBF+ architecture"
                if profile.profile_id == "son_hbf_plus" else "profile near-tier capacity"
            ),
            "hbf_write_parameters": (
                "unpublished: symmetric read/write bandwidth and 75 us program are synthetic sensitivity assumptions"
                if profile.bandwidth.get("hbf_write_gb_s") is None
                else "paper modeled write bandwidth and selected latency"
            ),
            "hbf_geometry": (
                "1.5 MiB migration block used as an effective service unit, not physical NAND geometry"
                if profile.profile_id == "hot_cold_b200"
                else "effective service lanes; physical channel/die geometry is not calibrated"
            ),
            "shared_secondary_tier": (
                "one synthetic GPU sees the shared eight-GPU tier without other GPU contention"
                if profile.hardware.get("gpu_count", 1) > 1 else None
            ),
        },
        "hbm_backend": "DramCore",
        "hbm_physical_config": asdict(parse_execution_physical_config(
            scenario.hardware.get_component("hbm0").metadata["physical_memory_config"])),
        "hbm_protocol": hbm_protocol_summary(parse_execution_physical_config(
            scenario.hardware.get_component("hbm0").metadata["physical_memory_config"])),
        "summary": {
            "makespan_ns": summary.get("makespan_ns"),
            "completed_requests": summary.get("completed_requests"),
            "rejected_requests": summary.get("rejected_requests"),
            "throughput_visible_output_tokens_per_s": throughput.get("visible_output_tokens_per_s"),
            "requests_per_s": throughput.get("requests_per_s"),
            "goodput_visible_output_tokens_per_s": goodput.get("visible_output_tokens_per_s") if slo_configured else None,
            "qualified_requests": goodput.get("qualified_requests") if slo_configured else None,
            "slo_configured": slo_configured,
            "ttft_ns": summary.get("ttft_ns"),
            "tpot_ns": summary.get("tpot_ns"),
            "e2e_ns": summary.get("e2e_ns"),
            "bottleneck_resource": summary.get("bottleneck_resource"),
        },
        "resources": {
            "resource_utilization": resource_utilization,
            "resource_breakdown": _resource_breakdown(resource_utilization),
            "gpu_resource_utilization": {
                key: value for key, value in resource_utilization.items()
                if key.startswith("gpu0.")
            },
            "gpu_resource_busy_ns": gpu_resource_busy,
            "gpu_busy_recorded_batches": gpu_busy_recorded_batches,
            "gpu_busy_scope": "per-resource occupied time recorded by persistent_live_kernel serving batches; GPU resources can overlap, so this is not wall-clock duration",
            "dram_traffic": {
                key: dram.get(key)
                for key in ("task_count", "logical_read_bytes", "logical_write_bytes", "physical_read_bytes",
                            "physical_write_bytes", "service_ns", "service_time_scope", "resource_totals",
                            "metric_availability", *DRAM_OPTIONAL_COUNTER_FIELDS)
            },
            "storage_traffic": {
                key: storage.get(key)
                for key in ("task_count", "logical_read_bytes", "logical_write_bytes", "service_ns")
                if key in storage
            },
        },
        "kv": {
            key: kv.get(key)
            for key in (
                "layout_mode", "capacity_bytes_by_component", "peak_bytes_by_component",
                "offload_events", "offload_bytes", "swap_events", "swap_bytes",
                "prefetch_events", "prefetch_bytes",
            )
            if key in kv
        },
        "trace": {
            "prompt_tokens_per_request": prompt_tokens,
            "output_tokens_per_request": output_tokens,
            "model": asdict(scenario.model),
            "request_count": len(result.serving.request_metrics),
            "batch_count": len(result.serving.batches),
            "scheduler": report.get("scheduler", {}),
        },
        "comparability": {
            "paper_tps_comparable": False,
            "reason": "full GPU/serving path is used, but paper-specific model, trace, multi-GPU topology and SLO workload are not fully public or matched; published compute is a scalar aggregate proxy on the bundled GPU model",
        },
    }


def run_profile(profile: CalibratedProfile, *,
                kv_pages: int = 8, steps: int = 4,
                latency_override_us: Optional[tuple[float, float]] = None) -> dict[str, Any]:
    if kv_pages < 2 or steps < 1:
        raise ValueError("kv_pages must be at least 2 and steps must be positive")
    hbm, hbf, applied = _profile_hardware(profile, latency_override_us=latency_override_us)
    traffic_page_bytes = hbf.config.page_bytes if hbf is not None else PAGE
    hbm_only = None
    if hbm is not None:
        hbm_only = _stream(
            hbm, None, kv_pages=kv_pages, steps=steps,
            kv_read_tier="hbm", kv_write_tier="hbm", weight_tier="hbm",
            traffic_page_bytes=traffic_page_bytes,
        )
    elif profile.profile_id == "son_hbf_plus":
        # HBF+ has no HBM stack in the paper configuration. Its separately
        # labelled HBM4 comparison uses the JEDEC reference organization.
        baseline = _hbm_core(
            capacity_bytes=288 * GB, read_gb_s=12800, write_gb_s=12800, stacks=8,
            profile_id="hbm4_reference",
        )
        hbm_only = _stream(
            baseline, None, kv_pages=kv_pages, steps=steps,
            kv_read_tier="hbm", kv_write_tier="hbm", weight_tier="hbm",
            traffic_page_bytes=traffic_page_bytes,
        )
        hbm_only["baseline_label"] = "son_hbm4_reference_not_same_profile"
        hbm_only["physical_config"] = asdict(baseline.config)
        hbm_only["hbm_protocol"] = hbm_protocol_summary(baseline.config)
    all_hbf = None
    if hbf is not None and profile.profile_id != "hot_cold_b200":
        # Each alternative starts with fresh resource queues.  Reusing the
        # baseline HBM core makes the HBF run wait for the preceding experiment.
        hbm, hbf, _ = _profile_hardware(profile, latency_override_us=latency_override_us)
        all_hbf = _stream(
            hbm, hbf, kv_pages=kv_pages, steps=steps,
            kv_read_tier="hbf", kv_write_tier="hbf",
            weight_tier="hbm" if hbm is not None else "hbf",
            traffic_page_bytes=traffic_page_bytes,
        )
    hot_cold = None
    if hbm is not None and hbf is not None:
        # Run with half the KV pages cold in HBF.  The cold tier is read on
        # the fixed stream, while generated KV remains an HBM write-on-evict
        # approximation for the mechanism comparison.
        hbm, hbf, _ = _profile_hardware(profile, latency_override_us=latency_override_us)
        cores = {"hbm": hbm, "hbf": hbf}
        by_tier: dict[str, list[Any]] = {"hbm": [], "hbf": []}
        clock = 0.0
        step_service_ns = []
        cold_pages = kv_pages // 2
        for step in range(steps):
            current = []
            weight = _request(hbm, operation=Operation.READ, address=max(16, kv_pages) * traffic_page_bytes, bytes_=8 * traffic_page_bytes, arrival_ns=clock, label=f"weight-{step}")
            by_tier["hbm"].append(weight); current.append(weight)
            for page in range(kv_pages):
                tier = "hbf" if page < cold_pages else "hbm"
                core = cores[tier]
                item = _request(core, operation=Operation.READ, address=page * traffic_page_bytes, bytes_=traffic_page_bytes, arrival_ns=clock, label=f"kv-read-{step}-{page}")
                by_tier[tier].append(item); current.append(item)
            for page in (kv_pages - 2, kv_pages - 1):
                item = _request(hbm, operation=Operation.WRITE, address=page * traffic_page_bytes, bytes_=traffic_page_bytes, arrival_ns=clock, label=f"kv-write-{step}-{page}")
                by_tier["hbm"].append(item); current.append(item)
            completion = max(item.completion_ns for item in current)
            step_service_ns.append(completion - clock)
            clock = completion
        hot_cold = {
            "placement_semantics": "static half-KV read partition; no session pause, write-on-evict, migration or prefetch is executed in this microbenchmark",
            "service_ns": float(clock),
            "service_us": float(clock / US),
            "step_service_ns": step_service_ns,
            "max_step_service_ns": max(step_service_ns),
            "logical_bytes": int(sum(item.logical_bytes for values in by_tier.values() for item in values)),
            "useful_bandwidth_gb_s": float(sum(item.logical_bytes for values in by_tier.values() for item in values) / clock) if clock else None,
            "tiers": {
                tier: {"logical_bytes": sum(item.logical_bytes for item in values), "logical_read_bytes": sum(item.logical_bytes for item in values if item.operation is Operation.READ), "logical_write_bytes": sum(item.logical_bytes for item in values if item.operation is Operation.WRITE)}
                for tier, values in by_tier.items()
            },
            "traffic": {"kv_pages": kv_pages, "cold_pages_on_hbf": cold_pages, "steps": steps},
        }
    tpot_s = profile.slo.get("threshold_s")
    if tpot_s is None:
        tpot_s = profile.slo.get("slo_tbt_s")
    for result in (hbm_only, all_hbf, hot_cold):
        if result is not None:
            result["slo_threshold_ns"] = None if tpot_s is None else float(tpot_s) * 1e9
            result["memory_service_slo_met"] = None if tpot_s is None else result["max_step_service_ns"] <= float(tpot_s) * 1e9
            result["slo_scope"] = "maximum single-step memory service only; excludes compute and serving queueing"
    return {
        "profile_id": profile.profile_id,
        "variant": "nominal",
        "latency_override_us": latency_override_us,
        "applied": applied,
        "hbm_only": hbm_only,
        "all_hbf": all_hbf,
        "hot_cold": hot_cold,
        "comparability": {
            "metric": "fixed logical memory service time and useful bandwidth",
            "full_paper_tps_comparable": False,
            "reason": "No full GPU compute, continuous batching, paper trace replay, or paper-specific SLO admission is in this backend entry point.",
        },
    }


def _ratio(numerator: Optional[dict[str, Any]], denominator: Optional[dict[str, Any]]) -> Optional[float]:
    if not numerator or not denominator or not denominator.get("service_ns"):
        return None
    return float(numerator["service_ns"] / denominator["service_ns"])


def _comparison_matrix(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix = []
    for result in results:
        profile_id = result["profile_id"]
        all_hbf = result.get("all_hbf")
        hbm = result.get("hbm_only")
        hot = result.get("hot_cold")
        matrix.append({
            "profile_id": profile_id,
            "variant": result.get("variant", "nominal"),
            "sim_hbf_vs_hbm_service_ratio": _ratio(all_hbf, hbm),
            "sim_hot_cold_vs_hbm_service_ratio": _ratio(hot, hbm),
            "full_backend": {
                "visible_output_tokens_per_s": (
                    result.get("full_backend", {})
                    .get("summary", {})
                    .get("throughput_visible_output_tokens_per_s")
                ),
                "goodput_visible_output_tokens_per_s": (
                    result.get("full_backend", {})
                    .get("summary", {})
                    .get("goodput_visible_output_tokens_per_s")
                ),
                "ttft_ns": result.get("full_backend", {}).get("summary", {}).get("ttft_ns"),
                "tpot_ns": result.get("full_backend", {}).get("summary", {}).get("tpot_ns"),
                "hbm_visible_output_tokens_per_s": (
                    result.get("full_backend", {})
                    .get("hbm_baseline", {})
                    .get("summary", {})
                    .get("throughput_visible_output_tokens_per_s")
                ),
                "hbf_vs_hbm_visible_tps_ratio": result.get("full_backend", {})
                .get("comparison", {})
                .get("hbf_vs_hbm_visible_tps_ratio"),
                "hbf_storage_task_count": result.get("full_backend", {})
                .get("resources", {})
                .get("storage_traffic", {})
                .get("task_count"),
                "hbm_storage_task_count": result.get("full_backend", {})
                .get("hbm_baseline", {})
                .get("resources", {})
                .get("storage_traffic", {})
                .get("task_count"),
                "paper_tps_comparable": result.get("full_backend", {})
                .get("comparability", {})
                .get("paper_tps_comparable", False),
            },
            "paper_reference": {
                "son_hbf": {"metric": "per-GPU batch gain", "reported_range": "1.3–4.5x", "status": "not_comparable"},
                "son_hbf_plus": {"metric": "per-GPU batch gain", "reported_range": "1.7–5.3x", "status": "not_comparable"},
                "hbf_sucks_hbf2_b200": {"metric": "E2E latency vs capacity-matched SSD", "reported_range": "2–5.5x", "status": "not_comparable"},
                "hot_cold_b200": {"metric": "resume overhead", "reported": "0.084 ms at N_a=48", "status": "weak_same_unit_only"},
            }.get(profile_id, {}),
        })
    return matrix


def build_report() -> dict[str, Any]:
    profiles = _profile_specs()
    results = []
    for profile in profiles:
        nominal = run_profile(profile)
        # Keep the physical memory stream and the full serving run separate:
        # the former isolates HBF service, while this run exercises the real
        # GPU/operator/scheduler path and reports its own comparability.
        direct_backend = _full_backend_summary(profile, use_hbf=True)
        hbm_backend = _full_backend_summary(profile, use_hbf=False)
        direct_summary = direct_backend.get("summary", {})
        hbm_summary = hbm_backend.get("summary", {})
        direct_backend["hbm_baseline"] = hbm_backend
        direct_backend["comparison"] = {
            "hbf_vs_hbm_visible_tps_ratio": (
                float(direct_summary.get("throughput_visible_output_tokens_per_s"))
                / float(hbm_summary.get("throughput_visible_output_tokens_per_s"))
                if hbm_summary.get("throughput_visible_output_tokens_per_s")
                else None
            ),
            "hbf_vs_hbm_goodput_ratio": (
                float(direct_summary.get("goodput_visible_output_tokens_per_s"))
                / float(hbm_summary.get("goodput_visible_output_tokens_per_s"))
                if hbm_summary.get("goodput_visible_output_tokens_per_s")
                else None
            ),
            "hbf_vs_hbm_ttft_p50_ratio": (
                float((direct_summary.get("ttft_ns") or {}).get("p50"))
                / float((hbm_summary.get("ttft_ns") or {}).get("p50"))
                if (hbm_summary.get("ttft_ns") or {}).get("p50")
                else None
            ),
            "hbf_vs_hbm_tpot_p50_ratio": (
                float((direct_summary.get("tpot_ns") or {}).get("p50"))
                / float((hbm_summary.get("tpot_ns") or {}).get("p50"))
                if (hbm_summary.get("tpot_ns") or {}).get("p50")
                else None
            ),
        }
        nominal["full_backend"] = direct_backend
        nominal["comparability"] = {
            **nominal["comparability"],
            "full_backend_used": True,
            "full_backend_metric": "visible output TPS, TTFT, TPOT, SLO goodput",
            "full_backend_paper_tps_comparable": False,
        }
        results.append(nominal)
        if profile.profile_id == "hbf_sucks_hbf2_b200":
            latency_sweep = []
            for read_us, write_us in profile.latency["sweep_us_read_write"]:
                sweep = run_profile(profile, latency_override_us=(read_us, write_us))
                latency_sweep.append({
                    "read_us": read_us,
                    "write_us": write_us,
                    "all_hbf_service_us": sweep["all_hbf"]["service_us"],
                    "all_hbf_vs_near_hbm_ratio": sweep["all_hbf"]["service_ns"] / sweep["hbm_only"]["service_ns"],
                })
            results.append({
                "profile_id": profile.profile_id,
                "experiment": "media_latency_sweep",
                "latency_sweep": latency_sweep,
                "paper_e2e_sensitivity": {"3.75x_media_speedup_e2e_change": "0.88% HBF-2; not directly comparable"},
            })
    # Hot-Cold's paper comparison is a block-fetch measurement.  Keep it in a
    # separate report entry so it is not confused with the fixed decode stream.
    hot = _profile_by_id("hot_cold_b200")
    block_results = []
    for blocks in (1, 48, 256):
        _, hbf, _ = _profile_hardware(hot)
        values = []
        for index in range(blocks):
            values.append(_request(hbf, operation=Operation.READ, address=index * hbf.config.page_bytes, bytes_=hbf.config.page_bytes, arrival_ns=0.0, label=f"resume-{blocks}-{index}"))
        completion = max(value.completion_ns for value in values)
        bytes_moved = sum(value.logical_bytes for value in values)
        block_results.append({"blocks": blocks, "bytes": bytes_moved, "resume_service_ns": completion, "resume_service_us": completion / US, "useful_bandwidth_gb_s": bytes_moved / completion})
    results.append({
        "profile_id": hot.profile_id,
        "experiment": "resume_block_fetch",
        "block_fetch": block_results,
        "paper_resume_overhead_us": hot.latency["resume_overhead_us_paper"],
        "paper_tbt_ms": hot.slo["tbt_s"] * 1000,
        "comparability": "same block/latency units, but missing-block count and overlap trace are not published",
    })
    return {
        "schema_version": 1,
        "description": "Paper-parameterized physical memory profiles plus a full V4 GPU/operator/CPU/serving probe with explicit non-memory policy; not full paper simulator reproduction.",
        "unit_conventions": {"bandwidth": "decimal GB/s", "latency": "ns", "capacity": "bytes/decimal GB unless profile says GiB/TiB"},
        "profiles": [profile.as_dict() for profile in profiles],
        "results": results,
        "comparison_matrix": _comparison_matrix([row for row in results if "profile_id" in row and "hbm_only" in row]),
        "execution_plan_status": {
            "scope": "This runner produces parameterized memory diagnostics and eight small serving runs; it does not execute the complete research/acceptance plan.",
            "executed": ["physical_memory_alternatives", "media_latency_sweep", "resume_block_fetch", "paired_full_backend_kv_probe"],
            "not_assessed_here": ["full_eight_experiment_matrix", "capacity_and_arrival_rate_sweeps", "write_buffer_acceptance", "session_lifecycle_acceptance", "package_tradeoff", "dense_and_moe_weight_placement", "direct_relay_dual_path", "named_GPU_validation", "paper_trace_reproduction"],
            "implementation_complete": None,
            "overall_plan_complete": None,
            "reason": "Completion and pass/fail must be established by the respective experiment and acceptance runners, never by static flags in this report.",
        },
        "limitations": [
            "The physical micro-profile does not include GPU compute or continuous batching; the separate full_backend result uses the V4 GPU/operator/CPU/scheduler path and reports grouped resource utilization.",
            "CIM is explicitly disabled for the calibrated memory-tier probe; enabling it remains a separate experiment because the papers do not provide a matching CIM workload or calibration.",
            "All HBM accesses use DramCore physical transactions; no aggregate-service execution mode is available.",
            "GPU compute and launch overhead retain the existing analytical operator/resource models; removing the HBM memory-service bypass does not make GPU execution cycle-accurate.",
            "HBF Sucks TokenSim traces, near-tier capacity tradeoff, queueing and SLO goodput are not reproduced.",
            "Hot-Cold original agent trajectories and simulator are not public; the block-fetch run is a mechanism check only.",
            "HBF+ serving retains an extra HBM support tier; it is a synthetic KV-placement probe, not the paper's HBF-only package.",
            "Hot-Cold writable serving uses explicit assumed write bandwidth/program latency; its published read-only resume diagnostic does not validate these assumptions.",
            "Full-backend TPS/TTFT/TPOT/SLO-goodput are reported for a small synthetic trace, but are not paper-comparable without the paper model, trace and topology.",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper-parameterized HBF diagnostics (not silicon calibration)",
        "",
        "These are physical memory-path profiles. They do not reproduce the papers' full LLM serving simulators.",
        "",
        "| profile | variant | experiment | service (us) | useful BW (GB/s) | HBF/HBM service ratio | SLO gate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        if "hbm_only" not in row:
            for item in row.get("block_fetch", []):
                lines.append(f"| {row['profile_id']} | — | resume {item['blocks']} blocks | {item['resume_service_us']:.3f} | {item['useful_bandwidth_gb_s']:.3f} | — | paper target {row['paper_resume_overhead_us']:.0f} us |")
            continue
        ratio = next((item["sim_hbf_vs_hbm_service_ratio"] for item in report["comparison_matrix"] if item["profile_id"] == row["profile_id"] and item["variant"] == row.get("variant", "nominal")), None)
        for name in ("hbm_only", "all_hbf", "hot_cold"):
            item = row.get(name)
            if not item:
                continue
            gate = item.get("memory_service_slo_met")
            lines.append(f"| {row['profile_id']} | {row.get('variant', 'nominal')} | {name} | {item['service_us']:.3f} | {item['useful_bandwidth_gb_s']:.3f} | {ratio:.3f} | {gate} |" if name == "all_hbf" and ratio is not None else f"| {row['profile_id']} | {row.get('variant', 'nominal')} | {name} | {item['service_us']:.3f} | {item['useful_bandwidth_gb_s']:.3f} | — | {gate} |")
    lines.extend([
        "",
        "## Full V4 backend probe",
        "",
        "This table uses the repository's GPU/operator cost model, continuous scheduler, HBM resource queue and SLO projection. The request trace is synthetic and intentionally small.",
        "",
        "| profile | variant | visible TPS | SLO goodput TPS | TTFT p50 (us) | TPOT p50 (us) | storage tasks | bottleneck |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in report["results"]:
        full = row.get("full_backend")
        if not full:
            continue
        variants = (("hbm_control", full.get("hbm_baseline")), ("hbf_direct", full))
        for variant, data in variants:
            if not data:
                continue
            summary = data.get("summary", {})
            ttft = (summary.get("ttft_ns") or {}).get("p50")
            tpot = (summary.get("tpot_ns") or {}).get("p50")
            bottleneck = summary.get("bottleneck_resource") or {}
            storage_tasks = (data.get("resources", {}).get("storage_traffic", {}) or {}).get("task_count", 0)
            lines.append(
                f"| {row['profile_id']} | {variant} | "
                f"{float(summary.get('throughput_visible_output_tokens_per_s') or 0):.2f} | "
                f"{('n/a' if not summary.get('slo_configured') else format(float(summary.get('goodput_visible_output_tokens_per_s') or 0), '.2f'))} | "
                f"{float(ttft or 0) / 1e3:.3f} | {float(tpot or 0) / 1e3:.3f} | "
                f"{int(storage_tasks or 0)} | {bottleneck.get('resource_id', 'unknown')} |"
            )
    lines.extend([
        "",
        "## Full-backend resource accounting",
        "",
        "GPU/CPU/互联使用现有算子与调度资源模型；CIM 在 HBF/HBM 内存路径实验中显式关闭，避免把未校准的 SRAM-CIM 开销混入结论。",
        "",
        "| profile | variant | GPU max util | CPU max util | CIM max util | interconnect max util | GPU calibration |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for row in report["results"]:
        full = row.get("full_backend")
        if not full:
            continue
        for variant, data in (("hbm_control", full.get("hbm_baseline")), ("hbf_direct", full)):
            if not data:
                continue
            breakdown = data.get("resources", {}).get("resource_breakdown", {})
            policy = data.get("backend_policy", {})
            lines.append(
                f"| {row['profile_id']} | {variant} | "
                f"{breakdown.get('gpu', {}).get('max_utilization', 0.0):.6f} | "
                f"{breakdown.get('cpu', {}).get('max_utilization', 0.0):.6f} | "
                f"{breakdown.get('cim', {}).get('max_utilization', 0.0):.6f} | "
                f"{breakdown.get('interconnect', {}).get('max_utilization', 0.0):.6f} | "
                f"{policy.get('gpu_compute_source', 'unknown')} |"
            )
    lines.extend([
        "",
        "## Paper comparison",
        "",
        "| profile | paper metric | simulator metric | comparability |",
        "|---|---|---|---|",
        "| Son HBF/HBF+ | per-GPU batch 1.3–4.5x / 1.7–5.3x | fixed memory service ratio | not comparable |",
        "| HBF Sucks | E2E latency 2–5.5x, goodput 1.1–2.7x lower | fixed memory service ratio | not comparable |",
        "| Hot-Cold | resume 0.084 ms at N_a=48 | block-fetch service in same units | weak same-unit comparison; missing-block trace is unpublished |",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
