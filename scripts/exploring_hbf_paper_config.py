"""Explicit catalog for Son et al., Exploring HBF (arXiv:2608.13868v1).

The catalog is separate from the existing calibrated-profile list and formal
scan. Builders expose controlled memory mechanisms, not a reproduction of the
paper's throughput. GB/MB are decimal; a NAND page is explicitly 4 KiB.

The NAND builder deliberately selects ``generic_nand`` page service. The paper
does not specify OCP, an FTL, or physical write-order/GC semantics. Applying the
simulator's OCP subset would add unreported assumptions. Actual die/plane
counts supply parallelism; no effective lane count is fitted to bandwidth.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.memory_types import DramConfig, NandConfig
from heterollm_sim.physical_contract import parse_execution_physical_config
from heterollm_sim.physical_presets import resolve_physical_config

GB = 1_000_000_000
MB = 1_000_000
PAGE_BYTES = 4096
PAPER_URL = "https://arxiv.org/html/2608.13868v1"
PAPER_DOI = "https://doi.org/10.1109/LCA.2026.3705817"
PAPER_ID = "son_exploring_hbf_2608_13868_v1"


def _integer(name: str, value: Any, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name: str, value: Any, *, positive: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")


@dataclass(frozen=True)
class PaperHardware:
    configuration_id: str
    hbm_stacks: int
    hbf_stacks: int
    total_capacity_gb: int
    hbm_read_gb_s: float
    hbm_write_gb_s: float
    hbf_read_gb_s: float
    hbf_write_gb_s: float
    dies_per_hbf_stack: int
    planes_per_hbf_die: int
    page_read_ns: float
    page_program_ns: float
    intermediate_storage: str
    intermediate_sram_bytes_per_stack: int
    read_staging_bytes_per_hbf_stack: int

    def __post_init__(self) -> None:
        if self.configuration_id not in {"HBM4", "CONV", "CONV+", "HBF", "HBF+"}:
            raise ValueError("unknown paper configuration_id")
        for name in ("hbm_stacks", "hbf_stacks", "dies_per_hbf_stack", "planes_per_hbf_die",
                     "intermediate_sram_bytes_per_stack", "read_staging_bytes_per_hbf_stack"):
            _integer(name, getattr(self, name), minimum=0)
        _integer("total_capacity_gb", self.total_capacity_gb)
        if self.hbm_stacks + self.hbf_stacks != 8:
            raise ValueError("paper configurations have eight total stacks")
        if self.total_capacity_gb != self.hbm_stacks * 36 + self.hbf_stacks * 512:
            raise ValueError("total capacity must equal the published decimal stack capacities")
        for name in ("hbm_read_gb_s", "hbm_write_gb_s", "hbf_read_gb_s", "hbf_write_gb_s",
                     "page_read_ns", "page_program_ns"):
            _number(name, getattr(self, name))
        for tier in ("hbm", "hbf"):
            present = getattr(self, tier + "_stacks") > 0
            for direction in ("read", "write"):
                rate = getattr(self, tier + "_" + direction + "_gb_s")
                if (rate > 0) != present:
                    raise ValueError(f"{tier} bandwidth must be positive exactly when that tier exists")
        if self.hbf_stacks:
            if (self.dies_per_hbf_stack != 16 or self.planes_per_hbf_die not in {16, 25}
                    or self.page_read_ns <= 0 or self.page_program_ns <= 0):
                raise ValueError("HBF requires the published die/plane organization and positive media timings")
        elif any((self.dies_per_hbf_stack, self.planes_per_hbf_die, self.page_read_ns,
                  self.page_program_ns, self.read_staging_bytes_per_hbf_stack)):
            raise ValueError("HBM-only configuration must not acquire NAND resources")
        if self.intermediate_storage not in {"HBM", "logic_die_SRAM"}:
            raise ValueError("unknown intermediate storage")
        if self.intermediate_storage == "logic_die_SRAM":
            if self.hbm_stacks or self.intermediate_sram_bytes_per_stack != 40 * MB:
                raise ValueError("plus configurations have no HBM and use 40 decimal MB SRAM per stack")
        elif not self.hbm_stacks or self.intermediate_sram_bytes_per_stack:
            raise ValueError("HBM intermediate storage requires HBM and no added logic SRAM")

    @property
    def hbm_capacity_bytes(self) -> int:
        return self.hbm_stacks * 36 * GB

    @property
    def hbf_capacity_bytes(self) -> int:
        return self.hbf_stacks * 512 * GB

    @property
    def nand_array_units(self) -> int:
        return self.hbf_stacks * self.dies_per_hbf_stack * self.planes_per_hbf_die

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "hbm_capacity_bytes": self.hbm_capacity_bytes,
                "hbf_capacity_bytes": self.hbf_capacity_bytes,
                "nand_array_units": self.nand_array_units,
                "intermediate_sram_bytes": self.hbf_stacks * self.intermediate_sram_bytes_per_stack,
                "read_staging_bytes": self.hbf_stacks * self.read_staging_bytes_per_hbf_stack,
                "source": "paper p2 Table I, Figure 2 and Section III",
                "gpu_core_reference": "NVIDIA DGX Rubin NVL8 [5]; arithmetic peak/efficiency not tabulated",
                "paper_workload_reproduced": False}


@dataclass(frozen=True)
class PaperWorkload:
    workload_id: str
    input_tokens: int
    output_tokens: int
    paper_input_label: str
    paper_output_label: str
    dataset: str
    length_basis: str

    def __post_init__(self) -> None:
        if self.workload_id not in {"SHORT", "MID", "LONG"}:
            raise ValueError("unknown paper workload_id")
        _integer("input_tokens", self.input_tokens)
        _integer("output_tokens", self.output_tokens)
        if not all(isinstance(getattr(self, name), str) and getattr(self, name).strip()
                   for name in ("paper_input_label", "paper_output_label", "dataset", "length_basis")):
            raise ValueError("workload labels and length basis must be non-empty strings")


def hardware_catalog() -> tuple[PaperHardware, ...]:
    """Five complete Table I configurations; no existing profile is changed."""
    return (
        PaperHardware("HBM4", 8, 0, 288, 12800., 12800., 0., 0., 0, 0, 0., 0., "HBM", 0, 0),
        PaperHardware("CONV", 1, 7, 3620, 1600., 1600., 2450., 73., 16, 16,
                      3000., 100000., "HBM", 0, 3_130_000),
        PaperHardware("CONV+", 0, 8, 4096, 0., 0., 2800., 84., 16, 16,
                      3000., 100000., "logic_die_SRAM", 40 * MB, 3_130_000),
        PaperHardware("HBF", 1, 7, 3620, 1600., 1600., 11200., 112., 16, 25,
                      1000., 100000., "HBM", 0, 3_130_000),
        PaperHardware("HBF+", 0, 8, 4096, 0., 0., 12800., 128., 16, 25,
                      1000., 100000., "logic_die_SRAM", 40 * MB, 3_130_000),
    )


def hardware_by_id(configuration_id: str) -> PaperHardware:
    for item in hardware_catalog():
        if item.configuration_id == configuration_id:
            return item
    raise ValueError("unknown paper configuration_id: " + str(configuration_id))


def workload_catalog() -> tuple[PaperWorkload, ...]:
    return (
        PaperWorkload("SHORT", 1660, 373, "1,660", "373", "shareGPT", "exact integer averages printed in paper"),
        PaperWorkload("MID", 5900, 499, "5.9K", "499", "LongBench",
                      "input K expanded as decimal 1000 for this catalog; paper gives a rounded average"),
        PaperWorkload("LONG", 103500, 1100, "103.5K", "1.1K", "English summarization / InfinityBench",
                      "K expanded as decimal 1000 for this catalog; paper gives rounded averages"),
    )


@dataclass(frozen=True)
class NandResearchAssumptions:
    """Undisclosed inputs must be supplied and recorded, never attributed to OCP."""
    pages_per_block: int
    blocks_per_plane: int
    block_erase_ns: float
    internal_bandwidth_gb_s: float
    max_outstanding_requests: int
    max_outstanding_write_requests: int | None
    host_full_duplex: bool
    host_granularity_bytes: int
    partial_page_policy: str
    basis: str

    def __post_init__(self) -> None:
        for name in ("pages_per_block", "blocks_per_plane", "max_outstanding_requests", "host_granularity_bytes"):
            _integer(name, getattr(self, name))
        for name in ("block_erase_ns", "internal_bandwidth_gb_s"):
            _number(name, getattr(self, name), positive=True)
        if self.max_outstanding_write_requests is not None:
            _integer("max_outstanding_write_requests", self.max_outstanding_write_requests)
            if self.max_outstanding_write_requests > self.max_outstanding_requests:
                raise ValueError("write request subset cannot exceed combined requests")
        if not isinstance(self.host_full_duplex, bool):
            raise ValueError("host_full_duplex must be boolean")
        if self.partial_page_policy not in {"reject", "read_modify_write"}:
            raise ValueError("partial_page_policy must be reject or read_modify_write")
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError("research assumption basis must be a non-empty string")


def default_research_assumptions(configuration_id: str) -> NandResearchAssumptions:
    """An explicit opt-in local mechanism control, not published hardware data.

    Whole-block coverage rounds up the array capacity without changing the
    Table I usable capacity, die count or plane count. This is not an FTL spare
    area/GC model or an inferred vendor organization.
    """
    hardware = hardware_by_id(configuration_id)
    if not hardware.hbf_stacks:
        raise ValueError("HBM4 has no NAND research assumptions")
    stripe = hardware.dies_per_hbf_stack * hardware.planes_per_hbf_die * 256 * PAGE_BYTES
    return NandResearchAssumptions(
        pages_per_block=256, blocks_per_plane=(512 * GB + stripe - 1) // stripe,
        block_erase_ns=2_000_000., internal_bandwidth_gb_s=100.,
        max_outstanding_requests=4096, max_outstanding_write_requests=None,
        host_full_duplex=True, host_granularity_bytes=PAGE_BYTES,
        partial_page_policy="read_modify_write",
        basis="Local mechanism control only: 256 pages/block, whole-block covering capacity, 2 ms erase, "
              "100 GB/s per die-channel, one aggregate full-duplex host link, 4096 global combined credits "
              "with writes sharing those credits, 4 KiB host-transfer granularity and paid read-modify-write "
              "for partial program pages. These are local choices, not parameters sourced from "
              "this paper; OCP is not selected.")


def build_paper_hbm_config(configuration_id: str) -> DramConfig | None:
    """Use a covering HBM4 standard reference, not an invented Rubin device."""
    hardware = hardware_by_id(configuration_id)
    if not hardware.hbm_stacks:
        return None
    config = resolve_physical_config("hbm4_reference", overrides={
        "stacks": hardware.hbm_stacks, "dies_per_stack": 16, "die_density_gbits": 32,
        "capacity_bytes": hardware.hbm_capacity_bytes,
        "interface_bandwidth_gb_s": hardware.hbm_read_gb_s,
        "read_bandwidth_gb_s": hardware.hbm_read_gb_s,
        "write_bandwidth_gb_s": hardware.hbm_write_gb_s,
        "metadata": {
            "paper_id": PAPER_ID, "paper_configuration_id": configuration_id,
            "source_url": PAPER_URL, "source_device_geometry_known": False,
            "capacity_input_scope": "Table I usable capacity in decimal GB",
            "model_scope": "controlled memory mechanism; not the paper serving workload",
            "research_assumptions": [
                "16H x 32 Gbit covering standard reference organization; not the undisclosed Rubin vendor part",
                "Table I aggregate bandwidth selected as a reference interface budget; vendor pin rate not disclosed",
                "HBM4 catalog reference normal-operation timing; not a calibrated Rubin speed bin"],
            "paper_workload_reproduced": False, "target_hardware_calibrated": False}})
    return parse_execution_physical_config(asdict(config))


def build_paper_nand_config(configuration_id: str, *,
                            assumptions: NandResearchAssumptions) -> NandConfig | None:
    """Construct genuine NandCore input with geometry-derived array resources.

    Native two-page bank buffers retain fill and reader lifetimes. They are
    media read buffers, not the separate 40-MB intermediate SRAM. This does
    not claim OCP control, a paper FTL or steady-state continuous batching.
    """
    hardware = hardware_by_id(configuration_id)
    if not isinstance(assumptions, NandResearchAssumptions):
        raise TypeError("assumptions must be explicit NandResearchAssumptions")
    if not hardware.hbf_stacks:
        return None
    cache_bytes_per_stack = (hardware.dies_per_hbf_stack
        * hardware.planes_per_hbf_die * 2 * PAGE_BYTES)
    buffer_basis = (
        "Two native page-buffer slots per actual die/plane. HBF: "
        "16 dies * 25 planes * 2 slots * 4096 bytes = 3276800 bytes "
        "= 3.125 MiB per stack; the paper's 3.13-MB label is interpreted "
        "as rounded binary MiB for this native organization, not 3130000 "
        "decimal bytes. CONV: 16 * 16 * 2 * 4096 = 2097152 bytes "
        "= 2 MiB per stack, below either interpretation of 3.13 MB. "
        "No extra staging arena or intermediate SRAM is created."
    )
    overrides = {
        "generation": "paper_HBF_projection", "protocol_mode": "generic_nand",
        "channels": hardware.hbf_stacks * hardware.dies_per_hbf_stack,
        "targets_per_channel": 1, "dies_per_target": 1, "luns_per_die": 1,
        "planes_per_lun": hardware.planes_per_hbf_die, "planes_independent": True,
        "parallel_units": None, "blocks_per_plane": assumptions.blocks_per_plane,
        "pages_per_block": assumptions.pages_per_block, "page_bytes": PAGE_BYTES,
        "host_granularity_bytes": assumptions.host_granularity_bytes, "internal_transfer_bytes": PAGE_BYTES,
        "capacity_bytes": hardware.hbf_capacity_bytes,
        "host_bandwidth_gb_s": max(hardware.hbf_read_gb_s, hardware.hbf_write_gb_s),
        "host_read_bandwidth_gb_s": hardware.hbf_read_gb_s,
        "host_write_bandwidth_gb_s": hardware.hbf_write_gb_s,
        "internal_bandwidth_gb_s": assumptions.internal_bandwidth_gb_s,
        "page_read_ns": hardware.page_read_ns, "page_program_ns": hardware.page_program_ns,
        "block_erase_ns": assumptions.block_erase_ns, "front_ns": 0.,
        "host_channels": 1, "host_full_duplex": assumptions.host_full_duplex,
        "read_cache_pages_per_bank": 2, "queue_scope": "device",
        "max_outstanding_requests": assumptions.max_outstanding_requests,
        "max_outstanding_write_requests": assumptions.max_outstanding_write_requests,
        "write_buffer_bytes": 0, "partial_page_policy": assumptions.partial_page_policy,
        "metadata": {
            "paper_id": PAPER_ID, "paper_configuration_id": configuration_id,
            "source_url": PAPER_URL, "source_section": "p2 Table I/Figure 2/Section III",
            "model_abstraction": "analytical_page_service",
            "native_memory_execution_contract": "native_nand_plane_double_buffer_v1",
            "execution_backend": "native_nand_core",
            "model_scope": "controlled NAND media mechanism; no OCP/FTL/GC or paper-serving reproduction",
            "geometry_status": "published_die_plane_counts_with_explicit_block_assumptions",
            "geometry_basis": "published stack/die/plane counts; local per-die channel mapping and block organization",
            "parallelism_basis": "stack count * 16 actual die resources * published planes/die; never bandwidth-fitted",
            "bandwidth_basis": "Table I aggregate directional ceilings; not a claim of measured sustained throughput",
            "timing_basis": "paper 1/3 us read and 100 us program; explicitly supplied local erase/front assumptions",
            "protocol_basis": "generic_nand; the article does not specify OCP",
            "paper_page_basis": "4 KiB read page is explicit; same program-page unit is a local modeling assumption",
            "host_granularity_basis": "explicit local transfer quantum, distinct from the published 4 KiB media page",
            "partial_page_basis": "explicit local policy; read_modify_write pays NandCore's old-page read, internal transfer "
                                  "and page program; reject disallows partial programs. Neither policy is given in the paper",
            "paper_read_staging_bytes": hardware.read_staging_bytes_per_hbf_stack * hardware.hbf_stacks,
            "paper_read_staging_execution": "native per-bank two-page read buffers; array, internal and host transfers "
                                            "retain independent resource calendars and live-reader overwrite dependencies; "
                                            "read requests drive fills, without an autonomous predictive prefetch controller",
            "native_read_cache_bytes_per_stack": cache_bytes_per_stack,
            "native_read_cache_bytes": cache_bytes_per_stack * hardware.hbf_stacks,
            "native_read_cache_unit_basis": buffer_basis,
            "native_read_cache_scope": "NandCore page-buffer lifetimes and paid physical transfers; "
                                       "not a separately modeled SRAM port or GPU activation workspace",
            "paper_intermediate_storage": hardware.intermediate_storage,
            "paper_intermediate_sram_bytes": hardware.intermediate_sram_bytes_per_stack * hardware.hbf_stacks,
            "paper_intermediate_sram_execution": "native SramCore and ordinary allocator/workspace lifetime; authored by exploring_hbf_native_config, not this NAND-only builder",
            "research_assumptions": asdict(assumptions),
            "additional_research_assumptions": [
                "one core channel per actual die; stack membership represented by consecutive groups of 16 channels",
                "one aggregate host link; explicit host-transfer granularity; zero fixed front delay",
                "two native read-cache pages per actual die/plane, no host-ack write staging; "
                "partial programs use the explicitly selected paid RMW/reject policy",
                buffer_basis,
                "generic direct-address page service; no automatic erases, write-order enforcement or GC",
                "whole-block covering capacity is not a disclosed reserve or overprovisioning ratio"],
            "paper_workload_reproduced": False, "target_hardware_calibrated": False}}
    config = resolve_physical_config("hbf_synthetic", overrides=overrides)
    return parse_execution_physical_config(asdict(config))


def execution_support(configuration_id: str) -> dict[str, Any]:
    """Describe native mechanisms separately from unpublished paper inputs."""
    hardware = hardware_by_id(configuration_id)
    mechanisms = ["native continuous batching and request/token scheduling",
                  "native TP/PP/EP lowering and topology-routed communication tasks",
                  "native physical allocations, workspace lifetime bounds and paged KV pools",
                  "native DRAM/NAND execution and shared memory/link resource calendars",
                  "native asynchronous KV page transfers with paid relay-memory reads and writes",
                  "native GPU kernel, attention working-set and shared-memory occupancy models",
                  "native external Prefill KV ingress with paid physical writes and admission after completion",
                  "native cohort-boundary warmup and measurement windows with fresh complete-request TPOT"]
    if hardware.hbf_stacks:
        mechanisms.append("native NAND two-page buffers per die/plane with fill and live-reader overwrite dependencies")
    if hardware.intermediate_storage == "logic_die_SRAM":
        mechanisms.append("native ordinary SramCore with finite pooled capacity, ports, queues and workspace lifetime")
    return {"configuration_id": configuration_id, "status": "available_with_research_assumptions",
            "paper_serving_supported": True, "memory_mechanism_builder_supported": True,
            "memory_mechanism_builder_status": "available_with_research_assumptions",
            "backend": "native_serving_physical", "mechanisms": mechanisms,
            "missing_native_mechanisms": [],
            "model_semantics": {
                "status": "native_semantics_implemented",
                "llama3_405b": "official BF16 KV8 tensor geometry mapped to the native graph and private rank-local storage",
                "llama4_maverick": "native fixed-chunk/full attention and KV rings, NoPE, optional temperature, sigmoid routing and expert input weighting; deterministic balanced expert histogram",
                "scope": "model-semantic work is separate from memory-builder and serving-controller availability"},
            "unresolved_inputs": ["paper effective GPU compute and exact kernels",
                                  "SRAM ports, NIC ownership, host ingress and duplex details",
                                  "request traces and measured expert routing",
                                  "original HBF simulator extension and raw result tables"],
            "scope": "new comparisons use exploring_hbf_native_config and native serving, planner and physical execution; "
                     "mechanism support does not assert numerical agreement with the paper or completed formal experiments; "
                     "historical operation/tile results remain historical bypass results and are not relabeled as native execution"}


def catalog_as_dict() -> dict[str, Any]:
    result = {
        "paper_id": PAPER_ID, "source_url": PAPER_URL, "source_doi": PAPER_DOI,
        "source_version": "arXiv v1, 14 August 2026; 4 pages, Table I only",
        "units": {"GB_bytes": GB, "MB_bytes": MB, "TB_s_bytes_per_second": 1000 * GB,
                  "bandwidth_field_units": "decimal GB/s", "latency_field_units": "ns",
                  "read_page_bytes": PAGE_BYTES},
        "hardware": [row.as_dict() for row in hardware_catalog()],
        "workloads": [asdict(row) for row in workload_catalog()],
        "serving": {"models": ["Llama 3 405B", "Llama 4 Maverick"],
                    "gpu_counts": [1, 2, 4, 8, 16], "decode_nodes_only": True,
                    "continuous_full_batch": True, "query_arrival_policy": "completion-matched steady state",
                    "default_tpot_slo_seconds": .1, "tpot_slo_sweep_seconds": [.05, .1, .2],
                    "offline_deadline_seconds": 24 * 3600,
                    "nvlink_gb_s": 1800, "gpus_per_node": 8, "infiniband_gb_s": 100,
                    "parallelism_search": ["DP", "TP", "PP", "EP"]},
        "execution_support": [execution_support(row.configuration_id) for row in hardware_catalog()],
        "undisclosed_paper_parameters": [
            "GPU arithmetic peak and achieved efficiency", "weight/KV precision",
            "exact full model configurations and per-operator kernels", "NAND blocks per plane and pages per block",
            "erase timing and endurance allocation policy", "FTL, GC and overprovisioning",
            "internal transfer topology/bandwidth and command front delay", "request admission credits and duplex policy",
            "exact interpretation of rounded K workload averages"],
        "comparison_scope": "comparison through native serving, planner and physical memory/link execution with native PD ingress, observation, SRAM and model semantics; "
                            "not yet a completed native paper experiment; historical bypass results retain their original identity, and existing scans are not restarted",
    }
    validate_catalog(result)
    return result


def validate_catalog(catalog: Mapping[str, Any] | None = None) -> None:
    """Reject non-finite values and malformed published configurations."""
    if catalog is None:
        for item in hardware_catalog():
            item.__post_init__()
        for item in workload_catalog():
            item.__post_init__()
        return
    if not isinstance(catalog, Mapping):
        raise ValueError("catalog must be a mapping")

    def walk(value: Any, path: str) -> None:
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(path + " must be finite")
        if isinstance(value, Mapping):
            for key, nested in value.items():
                walk(nested, path + "." + str(key))
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                walk(nested, path + f"[{index}]")

    walk(catalog, "catalog")
    hardware = catalog.get("hardware")
    workloads = catalog.get("workloads")
    if not isinstance(hardware, (list, tuple)) or len(hardware) != 5:
        raise ValueError("catalog must include all five Table I configurations")
    if not isinstance(workloads, (list, tuple)) or len(workloads) != 3:
        raise ValueError("catalog must include all three workloads")
    parsed_hardware = [PaperHardware(**{field.name: row[field.name] for field in fields(PaperHardware)})
                       for row in hardware]
    if len({row.configuration_id for row in parsed_hardware}) != 5:
        raise ValueError("duplicate hardware configuration")
    parsed_workloads = [PaperWorkload(**{field.name: row[field.name] for field in fields(PaperWorkload)})
                        for row in workloads]
    if len({row.workload_id for row in parsed_workloads}) != 3:
        raise ValueError("duplicate workload")


def export_catalog(path: str | Path) -> None:
    """Write catalog JSON only; no simulation or scan is invoked."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(catalog_as_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write structured paper catalog without running simulation")
    args = parser.parse_args()
    if args.output:
        export_catalog(args.output)
    else:
        print(json.dumps(catalog_as_dict(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
