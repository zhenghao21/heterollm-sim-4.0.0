"""Paper comparison input authoring for the normal native execution chain.

No operator timing, serving loop, capacity formula or throughput formula is
implemented here. Published interface targets and supplemental structural
assumptions are kept separate; every transfer is executed by the core.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from b200_hbm_hbf_scan_config import build_scan_scenario
from exploring_hbf_compare import _add_hbm_for_intermediates, _set_memory
from exploring_hbf_native_models import build_native_paper_model
from exploring_hbf_paper_config import (
    build_paper_hbm_config, build_paper_nand_config, default_research_assumptions,
    hardware_by_id, workload_catalog,
)
from heterollm_sim.component_presets import _gpu_cost_profile_template
from heterollm_sim.config import _gpu_profile_from_dict
from heterollm_sim.control_plane_planner import _derive_requirements
from heterollm_sim.cost_models import HostMemoryProfile
from heterollm_sim.ir import ComponentSpec, LinkSpec, ParallelSpec, PortSpec, RankMappingSpec, RequestSpec, model_graph_execution_layers
from heterollm_sim.memory_types import SramConfig
from heterollm_sim.physical_contract import require_physical_memory_config
from heterollm_sim.parallel import build_parallel_plan
from heterollm_sim.schema_v4 import ControllerProfile, GPUControllerProfile
from heterollm_sim.simulation_options import SimulationConfig

GPU_REFERENCE_ID = "rubin_public_peak_native_structural_reference_v1"
PD_BUFFER_BYTES = 64 * 1024**2
PHYSICAL_WINDOW_ASSUMPTION = (
    "Bounded physical execution has no sampled suffix: approximation error from forecasting is zero; "
    "this is not a claim of <10% hardware model error versus silicon or paper.")


def gpu_reference(component_id: str):
    """Use the original structural GPUProfile, not a scalar-cost bypass.

    Peak compute is public. SM/cache/issue organization below is an explicit
    research reference, not the undisclosed measured chip or paper kernels.
    """
    raw, _ = _gpu_cost_profile_template(component_id, sm_count=224,
        frequency_ghz=2., peak_bf16_tflops=4000.)
    raw.update(occupancy=1., attainable_efficiency=1., kernel_launch_ns=500.,
        scalar_ops_per_cycle=130000. / (224 * 128 * 2.), name=GPU_REFERENCE_ID)
    return _gpu_profile_from_dict(raw)


def sram_reference() -> SramConfig:
    return SramConfig(capacity_bytes=320_000_000, burst_bytes=64,
        interface_bandwidth_gb_s=80000., read_bandwidth_gb_s=80000.,
        write_bandwidth_gb_s=80000., read_latency_ns=1., write_latency_ns=1.,
        ports=8, max_outstanding_requests=256, metadata={
            "source_capacity": "paper: 8 stacks x 40 decimal MB per stack",
            "research_organization": "pooled 8-port byte-interleaved SRAM; stack-private arenas are not modeled",
            "research_timing": "10 TB/s per stack, 1 ns pipeline latency, 64B burst, 256 combined credits; not paper measurements"})


def _add_sram(template):
    config = sram_reference()
    owner = "sram0.sram_fabric"
    profile = HostMemoryProfile(bandwidth_gb_s=80000., efficiency=1.,
        read_bandwidth_gb_s=80000., write_bandwidth_gb_s=80000.,
        read_latency_ns=1., write_latency_ns=1., transaction_bytes=64,
        max_outstanding_requests=256, parallel_lanes=8, resource_id=owner,
        name="native logic-die SRAM research reference")
    port = PortSpec("host", "SRAM", role="endpoint", version="1", lanes=8,
                    bandwidth_gbps=640000., payload="streaming")
    sram = ComponentSpec("sram0", "sram", cost_profile_id="native-sram",
        ports=(port,), capacity_bytes=config.capacity_bytes, bandwidth_gbps=640000.,
        read_bandwidth_gbps=640000., write_bandwidth_gbps=640000.,
        metadata={"physical_memory_config": config, "memory_service_owner": owner,
                  "cost_profile_template": asdict(profile), "writable": True})
    components = tuple(replace(c, ports=c.ports + (replace(port, port_id="sram0", role="controller"),))
                       if c.component_id == "gpu0" else c for c in template.hardware.components)
    link = LinkSpec("gpu-sram", "gpu0", "sram0", "sram0", "host", "SRAM", "1", 8,
                    640000., 0., payload="streaming")
    profiles = {kind: dict(registry) for kind, registry in template.component_profiles.items()}
    profiles.setdefault("host_memory", {})["native-sram"] = profile
    return replace(template, hardware=replace(template.hardware, components=components + (sram,),
        links=template.hardware.links + (link,)), component_profiles=profiles)


def _renamed(value, names):
    if isinstance(value, str):
        for old, new in names.items():
            if value == old or value.startswith(old + ".") or value.startswith(old + ":"):
                return new + value[len(old):]
        return value
    if isinstance(value, dict):
        return {_renamed(key, names): _renamed(item, names) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_renamed(item, names) for item in value)
    if isinstance(value, list):
        return [_renamed(item, names) for item in value]
    return value


def build_native_comparison_scenario(model_id: str, hardware_id: str, workload_id: str,
        *, batch_size: int = 1, tp: int = 8, pp: int = 1, ep: int = 1,
        model_override=None, prompt_override=None, output_override=None,
        warmup_override=None, measurement_override=None):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    published = hardware_by_id(hardware_id)
    workload = next((row for row in workload_catalog() if row.workload_id == workload_id), None)
    if workload is None:
        raise ValueError("unknown workload")
    model = model_override if model_override is not None else build_native_paper_model(model_id)
    prompt = workload.input_tokens if prompt_override is None else prompt_override
    output = workload.output_tokens if output_override is None else output_override
    if prompt < 2 or output < 2 or prompt + output > model.max_sequence_length:
        raise ValueError("invalid request length or native model context exceeded")
    world = tp * pp * ep
    if min(tp, pp, ep) < 1 or world > 16:
        raise ValueError("native deployment must use 1..16 ranks")
    # The old scan builder supplies only CPU/HOST wiring. Discard its model,
    # tensor targets, source identity and rank contracts below.
    template = build_scan_scenario("qwen2_5-0_5b", "HBM" if hardware_id == "HBM4" else "HBF", 2, 2)
    if hardware_id != "HBM4" and published.hbm_stacks:
        template = _add_hbm_for_intermediates(template)
    if published.hbm_stacks:
        template = _set_memory(template, "hbm0", build_paper_hbm_config(hardware_id))
    if published.hbf_stacks:
        template = _set_memory(template, "hbf0", build_paper_nand_config(hardware_id,
            assumptions=default_research_assumptions(hardware_id)))
    if not published.hbm_stacks:
        template = _add_sram(template)
    active0 = "hbm0" if published.hbm_stacks else "sram0"
    weight0 = "hbf0" if published.hbf_stacks else "hbm0"
    components = [replace(c, ports=tuple(replace(port, max_links=max(world, port.max_links))
        if c.component_id == "cpu0" and port.port_id == "pcie0" else port for port in c.ports))
        for c in template.hardware.components if c.component_id in {"cpu0", "hostmem0"}]
    links = [deepcopy(link) for link in template.hardware.links
             if {link.source_component, link.target_component} <= {"cpu0", "hostmem0"}]
    registries = {kind: {key: value for key, value in registry.items()
                        if key in {c.cost_profile_id for c in components}}
                  for kind, registry in template.component_profiles.items()}
    local = [c for c in template.hardware.components if c.component_id not in {"cpu0", "hostmem0"}]
    ranks = []
    layers = model_graph_execution_layers(model.graph)
    for index in range(world):
        names = {"gpu0": f"gpu{index}", "hbm0": f"hbm{index}",
                 "hbf0": f"hbf{index}", "sram0": f"sram{index}"}
        for c in local:
            cid = names[c.component_id]
            profile_id = "native-" + cid
            metadata = _renamed(deepcopy(c.metadata), names)
            if c.component_id == "gpu0":
                profile = gpu_reference(cid)
                metadata = {"native_gpu_reference": GPU_REFERENCE_ID,
                    "public_peaks": {"bf16_dense_tflops": 4000., "fp32_vector_tflops": 130.,
                        "source": "https://www.nvidia.com/en-us/data-center/hgx/"},
                    "research_structure": "224 SM, 2GHz, four MMA units/SM with cycles matching public peak; 30MiB L1/shared, 50MiB L2 reference, not vendor calibration",
                    "cost_profile_template": asdict(profile)}
                kind = "gpu"
                ports = c.ports + (PortSpec("peer", "NVLink", "peer", version="6.0", lanes=1,
                    bandwidth_gbps=14400., max_links=2, payload="streaming"),
                    PortSpec("pd", "InfiniBand", "peer", version="1.0", lanes=1,
                    bandwidth_gbps=800., max_links=2, payload="streaming"))
            else:
                kind = template.component_profile_kind(c)
                raw = _renamed(asdict(template.resolve_component_profile(c)), names)
                profile = type(template.resolve_component_profile(c))(**raw)
                metadata["cost_profile_template"] = asdict(profile)
                ports = c.ports
                if metadata.get("physical_memory_config") is not None:
                    physical = require_physical_memory_config(c)
                    # Preserve typed geometry; only bind private resource IDs.
                    metadata["physical_memory_config"] = replace(physical,
                        metadata=_renamed(deepcopy(physical.metadata), names))
            registries.setdefault(kind, {})[profile_id] = profile
            components.append(replace(c, component_id=cid, cost_profile_id=profile_id,
                package_id=f"native-package-{index}", die_id=f"native-{cid}", ports=ports, metadata=metadata))
        for link in template.hardware.links:
            if not (link.source_component in names or link.target_component in names):
                continue
            links.append(replace(link, link_id=f"rank{index}-" + link.link_id,
                source_component=names.get(link.source_component, link.source_component),
                target_component=names.get(link.target_component, link.target_component),
                metadata=_renamed(deepcopy(link.metadata), names)))
        ranks.append(RankMappingSpec(index, f"gpu{index}", index % tp,
            index // (tp * ep), (index // tp) % ep, memory_component_id=names[active0]))
    for a in range(world):
        for b in range(a + 1, world):
            # Eight-GPU node: a ring gives each GPU two directional 900 GB/s
            # ports. No all-to-all collection of free full-bandwidth links.
            if a // 8 == b // 8 and (b == a + 1 or (a % 8 == 0 and b == min(world - 1, a + 7))):
                links.append(LinkSpec(f"nvlink-{a}-{b}", f"gpu{a}", "peer", f"gpu{b}", "peer",
                    "NVLink", "6.0", 1, 7200., 500., payload="streaming"))
            elif world > 8 and a % 8 == b % 8 and b == a + 8:
                links.append(LinkSpec(f"ib-node-{a}-{b}", f"gpu{a}", "pd", f"gpu{b}", "pd",
                    "InfiniBand", "1.0", 1, 800., 500., payload="streaming"))
    host = template.hardware.get_component("hostmem0")
    host_profile = template.resolve_component_profile(host)
    pd_physical = require_physical_memory_config(host)
    pd_physical = replace(pd_physical, capacity_bytes=PD_BUFFER_BYTES,
        interface_bandwidth_gb_s=200., read_bandwidth_gb_s=200., write_bandwidth_gb_s=200.,
        metadata={**pd_physical.metadata, "research_scope": "external Prefill producer source, bounded streaming buffer; compute excluded"})
    pd_profile = replace(host_profile, bandwidth_gb_s=200., read_bandwidth_gb_s=200.,
        write_bandwidth_gb_s=200., resource_id="pd_host.memory")
    registries.setdefault("host_memory", {})["native-pd-host"] = pd_profile
    components.append(replace(host, component_id="pd_host", cost_profile_id="native-pd-host",
        capacity_bytes=PD_BUFFER_BYTES, bandwidth_gbps=1600., read_bandwidth_gbps=1600., write_bandwidth_gbps=1600.,
        ports=(PortSpec("network", "InfiniBand", "peer", version="1.0", lanes=1,
            bandwidth_gbps=800. * world, max_links=world, payload="streaming"),),
        metadata={"physical_memory_config": pd_physical, "memory_service_owner": "pd_host.memory",
                  "cost_profile_template": asdict(pd_profile)}))
    for index in range(world):
        links.append(LinkSpec(f"pd-ingress-{index}", "pd_host", "network", f"gpu{index}", "pd",
            "InfiniBand", "1.0", 1, 800., 500., payload="streaming"))
    hardware = replace(template.hardware, name=f"native-paper-{hardware_id}-{world}gpu",
        components=tuple(components), links=tuple(links), metadata={"native_comparison": True,
        "gpu_reference": GPU_REFERENCE_ID, "network": "per-node ring: 900GB/s per direction per adjacent edge, two outgoing edges/GPU; 100GB/s PD links; link startups 500ns research",
        "paper_values_fitted": False})
    parallel = ParallelSpec(tp_degree=tp, pp_degree=pp, ep_degree=ep, rank_mapping=tuple(ranks),
        layer_to_stage={layer.layer_id: min(pp - 1, index * pp // len(layers)) for index, layer in enumerate(layers)},
        collective_algorithm="ring")
    warmup = output - 1 if warmup_override is None else warmup_override
    measurement = 2 * (output - 1) if measurement_override is None else measurement_override
    scheduler = replace(template.workload.scheduler, mode="continuous", max_num_seqs=batch_size,
        max_num_batched_tokens=batch_size, max_num_ubatch_tokens=batch_size,
        prefill_chunk_tokens=1, preemption_enabled=False, slo_ttft_ns=None, slo_tbt_ns=None,
        slo_tpot_ns=100_000_000.)
    rank_stores = {str(index): (f"hbf{index}" if published.hbf_stacks else f"hbm{index}") for index in range(world)}
    metadata = {"rank_weight_components": rank_stores, "rank_kv_components": rank_stores,
        "llama_backend_memory": {f"hbf{i}": {"device_id": f"gpu{i}", "buffer_kind": "HBF", "access": "direct", "priority": 1}
                                 for i in range(world)} if published.hbf_stacks else {}}
    requests = []
    for index in range(batch_size * 4):
        # Paid warmup seeds spread completion phases across slots. These are
        # actual longer histories with fewer remaining visible outputs, not
        # free generated tokens or a hand-written steady-state scheduler.
        age = index * (output - 1) // batch_size if index < batch_size else 0
        requests.append(RequestSpec(f"native-request-{index:08d}", 0., prompt + age, output - age,
                                    metadata={"warmup_seed": index < batch_size}))
    s = replace(template, name=f"native-{model_id}-{hardware_id}-{workload_id}-B{batch_size}",
        hardware=hardware, component_profiles=registries, model=model,
        simulation=SimulationConfig(mode="detailed"),
        fusion_policy=replace(template.fusion_policy, flash_attention=True),
        runtime_profile=ControllerProfile(gpu_controllers={f"gpu{i}": GPUControllerProfile() for i in range(world)}),
        placement=replace(template.placement, hardware_name=hardware.name, model_name=model.name,
            parallel=parallel, tensor_to_component={}, tensor_bytes={}, op_to_component={}, metadata=metadata,
            kv_policy=replace(template.placement.kv_policy, cache_component=weight0, allocation_policy="lazy", dtype="bf16")),
        workload=replace(template.workload, name="native-decode-node-continuous",
            requests=tuple(requests),
            request_count=batch_size * 4, scheduler=scheduler, metadata={"native_paper_comparison": True,
                "external_kv_ingress": {"source_component": "pd_host", "buffer_bytes": PD_BUFFER_BYTES,
                    "chunk_bytes": 1024**2, "max_inflight_requests": 2, "relay_components": [],
                    "require_full_batch": True, "warmup_decode_cohorts": warmup,
                    "measurement_decode_cohorts": measurement, "finish_after_measurement": True}}),
        assumptions=("Native serving, planner, GPU cost profiles and physical DRAM/NAND/SRAM services; no paper_* execution.",
            "Paper Table I per-GPU capacity and interface budgets, BF16 model and KV, decimal workload K.",
            "Native generic NAND page service; unspecified block geometry and timing are research assumptions, not OCP or vendor GC.",
            "8x40MB intermediate SRAM is modeled as a pooled 8-port array; bandwidth/latency/credits are explicit research inputs.",
            "External producer streams prompt-1 KV tokens; native one-token final-prompt generates the first visible output and pays its complete cost.",
            "Warmup seeds use real age-extended prompt histories and shorter remaining outputs to stagger completion; fresh measured requests retain the declared paper lengths.",
            PHYSICAL_WINDOW_ASSUMPTION))
    requirements = _derive_requirements(s)
    native_parallel = build_parallel_plan(s)
    operator_targets = {}
    for item in requirements:
        if not item.mapping_key or item.state_tensor:
            continue
        if item.kind == "embedding":
            stage = 0
        elif item.kind == "lm_head" or item.kind.startswith("mtp_"):
            stage = pp - 1
        else:
            stage = native_parallel.stage_for_layer(item.layer) if item.layer is not None else 0
        operator_targets[item.mapping_key] = native_parallel.rank_at(0, stage, 0).component_id
    options = {"design_decode_batch_size": batch_size, "design_prefill_tokens": 1,
        "operator_targets": operator_targets,
        "kv_cache_target": weight0}
    return replace(s, placement=replace(s.placement,
        metadata={**s.placement.metadata, "control_plane": {"policy": {"options": options}}}))
