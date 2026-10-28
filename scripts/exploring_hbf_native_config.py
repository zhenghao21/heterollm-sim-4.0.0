"""Paper comparison input authoring for the normal native execution chain.

No operator timing, serving loop, capacity formula or throughput formula is
implemented here. Published interface targets and supplemental structural
assumptions are kept separate; every transfer is executed by the core.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import math
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
    "Bounded physical execution has no sampled suffix; the fast path reports its conditional sampling interval "
    "separately and does not use external hardware verification as an acceptance target.")
PD_LINK_BYTES_PER_SECOND = 100e9
CONTROL_PLANE_TIME_LIMIT_S = 3600.
_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}

# Approved PD controller-state approximations for these scenario/layouts,
# consistently across all batch sizes searched there.
_NATIVE_FAST_WINDOW_OVERRIDES = {
    ("llama3_405b", "HBM4", "SHORT", 8, 1, 1): {"tp_rank_symmetry": "all"},
    ("llama4_maverick", "HBM4", "SHORT", 2, 1, 4): {"tp_rank_symmetry": "all"},
    ("llama4_maverick", "HBM4", "MID", 4, 1, 2): {"tp_rank_symmetry": "all"},
}


def native_fast_window_options(model_id, hardware_id, workload_id, *,
                               tp, pp, ep, max_wall_seconds):
    """Resolve the same per-layout fast options for manager and stress runs."""
    from heterollm_sim.simulation_options import NativeWindowFastConfig

    key = (model_id, hardware_id, workload_id, tp, pp, ep)
    return NativeWindowFastConfig(max_wall_seconds=max_wall_seconds,
                                 **_NATIVE_FAST_WINDOW_OVERRIDES.get(key, {}))


def native_import_pool(layers, prompt, output, batch_size, tp, pp, ep, published, slo_tpot_s=.1):
    """Prepared-import slots for continuous refill near the TPOT SLO.

    Every EP group owns B/ep requests and refills only its own slots, so the
    pool is per group: each step frees up to ceil((B/ep)/output) slots of a
    group at once, each refilled at the next step from one of that group's
    ready requests. A request entering the pool must finish its import, at the
    per-rank PD link and KV-tier write rates of Table I, before its turn: the
    pool holds that many requests for every step an import may span, with a
    factor of four margin (twice the import time at half the SLO TPOT). A
    request's KV lives only on its own group's tp*pp ranks, hence the import
    time divides by tp*pp, not by ep. The pool's KV is ordinary allocated
    capacity, so it counts against the batch. Returns the total over groups.
    """
    if batch_size % ep:
        raise ValueError("batch_size must be divisible by ep")
    kv_bytes = 0
    for layer in layers:
        if layer.sequence_mixer == "linear_attention" or not layer.kv_heads:
            continue
        head_dim = layer.attention_head_dim or layer.hidden_size // layer.attention_heads
        tokens = min(prompt-1, layer.attention_chunk_size) if layer.attention_chunk_size else prompt-1
        kv_bytes += 2 * layer.kv_heads * head_dim * _DTYPE_BYTES[layer.dtype] * tokens
    write = (published.hbf_write_gb_s if published.hbf_stacks else published.hbm_write_gb_s) * 1e9
    seconds = kv_bytes / (tp * pp) / min(PD_LINK_BYTES_PER_SECOND, write)
    return ep * math.ceil(batch_size // ep / output) * (1 + math.ceil(4 * seconds / slo_tpot_s))


def native_capacity_batch(requests_per_group, batch_size, pool, output, ep):
    """Largest per-replica batch whose seeds and import pool fit the KV a group held.

    Every seed and every prepared import reserves the same max-live KV
    (prompt + output - 1 tokens), so a group that held ``requests_per_group``
    when KV refused one more fits B/ep seeds plus its pool share. The pool's
    depth per freed slot is read off ``pool``, sized by ``native_import_pool``
    for ``batch_size``; it does not depend on the batch. Returns 0 if not
    even one request per group fits.
    """
    depth = pool // (ep * math.ceil(batch_size // ep / output))
    group = requests_per_group
    while group > 0 and group + math.ceil(group / output) * depth > requests_per_group:
        group -= 1
    return group * ep


def validate_native_layout(layers, batch_size, ep):
    """Reject an EP degree the model or batch cannot be split by.

    The batch is per data-parallel replica; its B/ep requests per EP group
    and every MoE layer's experts must divide evenly. EP shards the experts
    of MoE layers only, so a model without any has nothing to split.
    """
    if batch_size % ep:
        raise ValueError(f"batch_size {batch_size} must be divisible by ep {ep}")
    if ep == 1:
        return
    moe = [layer for layer in layers if layer.is_moe]
    if not moe:
        raise ValueError("ep > 1 requires a model with MoE layers")
    bad = sorted({layer.num_experts for layer in moe if layer.num_experts % ep})
    if bad:
        raise ValueError(f"every MoE layer's num_experts must be divisible by ep {ep}; got {bad}")


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
        *, batch_size: int = 1, tp: int = 8, pp: int = 1, ep: int = 1, pipeline_microbatches: int = 1,
        model_override=None, prompt_override=None, output_override=None,
        warmup_override=None, measurement_override=None, import_pool=None):
    """Build one data-parallel replica of the native paper deployment.

    The replica spans tp*pp*ep GPUs and ``batch_size`` is its batch, shared
    by its ep attention-DP groups (B/ep requests each); independent replicas
    (DP) are not simulated here, the runner scales one replica's throughput.
    ``import_pool`` replaces the Table-I prepared-import pool with the one a
    native run's measured import span needs (``native_stationary``).
    ``pipeline_microbatches`` partitions each group's request slots inside
    one cohort; it does not change the logical batch, Prefill chunk or token
    ubatch limits. The next cohort starts after this pipeline has drained.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if type(pipeline_microbatches) is not int or pipeline_microbatches < 1:
        raise ValueError("pipeline_microbatches must be positive")
    if import_pool is not None and (isinstance(import_pool, bool) or not isinstance(import_pool, int)
                                    or import_pool < ep or import_pool % ep):
        raise ValueError("import_pool must be a positive multiple of ep")
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
    if batch_size % (ep * pipeline_microbatches):
        raise ValueError("batch_size must be divisible by ep * pipeline_microbatches")
    layers = model_graph_execution_layers(model.graph)
    validate_native_layout(layers, batch_size, ep)
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
    pd_source = require_physical_memory_config(host)
    # The external Prefill producer is a streaming source for all world PD
    # links.  Keep the native DRAM service, but expand its channel geometry so
    # the physical source peak is at least the aggregate link rate.  The
    # per-channel DDR5 timing and burst interval remain those of the source
    # module; no constant-bandwidth shortcut is introduced.
    channel_peak_gb_s = float(pd_source.interface_bandwidth_gb_s) / pd_source.channels
    target_gb_s = world * PD_LINK_BYTES_PER_SECOND / 1e9
    channels = max(pd_source.channels, math.ceil(target_gb_s / channel_peak_gb_s))
    pd_bandwidth_gb_s = channel_peak_gb_s * channels
    # Every prepared import owns a PD_BUFFER_BYTES staging buffer (its chunk
    # slots), so the producer holds one per pool entry.
    pool = (native_import_pool(layers, prompt, output, batch_size, tp, pp, ep, published)
            if import_pool is None else import_pool)
    pd_physical = replace(pd_source, capacity_bytes=pool * PD_BUFFER_BYTES, channels=channels,
        interface_bandwidth_gb_s=pd_bandwidth_gb_s, read_bandwidth_gb_s=pd_bandwidth_gb_s,
        write_bandwidth_gb_s=pd_bandwidth_gb_s,
        metadata={**pd_source.metadata, "research_scope": "external Prefill producer source, bounded streaming buffer; compute excluded",
                  "known_values": "%d channels x 5600 MT/s x 64-bit payload = %.1f GB/s aggregate" % (channels, pd_bandwidth_gb_s),
                  "source_bandwidth_basis": "native DDR5 geometry expanded to %d channels at %.1f GB/s per channel; target world x %.1f GB/s PD links" %
                  (channels, channel_peak_gb_s, PD_LINK_BYTES_PER_SECOND / 1e9),
                  "source_bandwidth_gb_s": pd_bandwidth_gb_s,
                  "geometry_status": "explicit_assumption",
                  "geometry_basis": "native DDR5 channel organization expanded to satisfy aggregate PD source demand"})
    pd_profile = replace(host_profile, bandwidth_gb_s=pd_bandwidth_gb_s,
        read_bandwidth_gb_s=pd_bandwidth_gb_s, write_bandwidth_gb_s=pd_bandwidth_gb_s,
        transaction_bytes=pd_physical.burst_bytes, parallel_lanes=channels,
        resource_id="pd_host.memory")
    registries.setdefault("host_memory", {})["native-pd-host"] = pd_profile
    components.append(replace(host, component_id="pd_host", cost_profile_id="native-pd-host",
        capacity_bytes=pool * PD_BUFFER_BYTES, bandwidth_gbps=pd_bandwidth_gb_s * 8,
        read_bandwidth_gbps=pd_bandwidth_gb_s * 8, write_bandwidth_gbps=pd_bandwidth_gb_s * 8,
        ports=(PortSpec("network", "InfiniBand", "peer", version="1.0", lanes=1,
            bandwidth_gbps=pd_bandwidth_gb_s * 8, max_links=world, payload="streaming"),),
        metadata={"physical_memory_config": pd_physical, "memory_service_owner": "pd_host.memory",
                  "cost_profile_template": asdict(pd_profile)}))
    for index in range(world):
        links.append(LinkSpec(f"pd-ingress-{index}", "pd_host", "network", f"gpu{index}", "pd",
            "InfiniBand", "1.0", 1, 800., 500., payload="streaming"))
    hardware = replace(template.hardware, name=f"native-paper-{hardware_id}-{world}gpu",
        components=tuple(components), links=tuple(links), metadata={"native_comparison": True,
        "gpu_reference": GPU_REFERENCE_ID, "network": "per-node ring: 900GB/s per direction per adjacent edge, two outgoing edges/GPU; 100GB/s PD links; link startups 500ns research",
        "paper_values_fitted": False})
    parallel = ParallelSpec(tp_degree=tp, pp_degree=pp, ep_degree=ep,
        pipeline_microbatches=pipeline_microbatches, rank_mapping=tuple(ranks),
        layer_to_stage={layer.layer_id: min(pp - 1, index * pp // len(layers)) for index, layer in enumerate(layers)},
        collective_algorithm="ring")
    warmup = output - 1 if warmup_override is None else warmup_override
    measurement = 2 * (output - 1) if measurement_override is None else measurement_override
    # Continuous batching: a freed slot refills at the next step from the
    # prepared-import pool, and each new request's one-token final prompt
    # rides in that mixed decode cohort instead of a separate forward pass.
    scheduler = replace(template.workload.scheduler, mode="continuous", max_num_seqs=batch_size,
        max_num_batched_tokens=batch_size, max_num_ubatch_tokens=batch_size,
        prefill_chunk_tokens=1, mixed_phase_batching=True, preemption_enabled=False,
        slo_ttft_ns=None, slo_tbt_ns=None, slo_tpot_ns=100_000_000.)
    rank_stores = {str(index): (f"hbf{index}" if published.hbf_stacks else f"hbm{index}") for index in range(world)}
    metadata = {"rank_weight_components": rank_stores, "rank_kv_components": rank_stores,
        "llama_backend_memory": {f"hbf{i}": {"device_id": f"gpu{i}", "buffer_kind": "HBF", "access": "direct", "priority": 1}
                                 for i in range(world)} if published.hbf_stacks else {}}
    requests = []
    for index in range(batch_size * 4):
        # Warmup seeds spread completion phases across slots. These are
        # actual longer histories with fewer remaining visible outputs, not
        # free generated tokens or a hand-written steady-state scheduler.
        # Their KV is preloaded (allocator and capacity charged, no paid
        # import); every fresh request still pays its native PD import.
        # Centered ages cover all output phases 0..output-1, so the first
        # step's mean context is the steady state's.
        age = (2 * index + 1) * output // (2 * batch_size) if index < batch_size else 0
        requests.append(RequestSpec(f"native-request-{index:08d}", 0., prompt + age, output - age,
                                    metadata={"warmup_seed": index < batch_size}))
    s = replace(template, name=f"native-{model_id}-{hardware_id}-{workload_id}-TP{tp}PP{pp}EP{ep}-B{batch_size}",
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
                    "chunk_bytes": 1024**2, "max_inflight_requests": pool, "relay_components": [],
                    "require_full_batch": False, "preload_warmup_seed_kv": True,
                    "warmup_decode_cohorts": warmup,
                    "measurement_decode_cohorts": measurement, "finish_after_measurement": True}}),
        assumptions=("Native serving, planner, GPU cost profiles and physical DRAM/NAND/SRAM services; no paper_* execution.",
            "Paper Table I per-GPU capacity and interface budgets, BF16 model and KV, decimal workload K.",
            "Native generic NAND page service; unspecified block geometry and timing are research assumptions, not OCP or vendor GC.",
            "8x40MB intermediate SRAM is modeled as a pooled 8-port array; bandwidth/latency/credits are explicit research inputs.",
            "External Prefill producer source uses native DDR5 service geometry expanded to at least world x 100 GB/s; PD links and target KV writes remain the intended bottlenecks (research assumption).",
            "PD import (user approved 2026-10-24): one stream per target rank, all streams concurrent; each import owns a %d MiB source buffer cut into 1 MiB chunk slots dealt round-robin to its streams; a chunk holds its slot from source read until its target write completes (RDMA source-buffer semantics), so a stream keeps at most its slot count in flight while adjacent chunks' reads, links and writes overlap on the native timelines." % (PD_BUFFER_BYTES // 1024**2),
            "External producer streams prompt-1 KV tokens; native one-token final-prompt generates the first visible output and pays its complete cost.",
            "Warmup seeds use real age-extended prompt histories and shorter remaining outputs to stagger completion; fresh measured requests retain the declared paper lengths.",
            "Warmup-seed KV is preloaded with allocator pages and capacity charged but no paid import; controllers start clean with no in-flight writes (the paper does not publish its initial state). Every fresh request pays its native PD import.",
            "Continuous batching: freed slots refill at the next step from the pre-imported requests of their own EP group (%d group(s), up to %d per group, %d in total; pool sized %s; its KV counts against capacity); final-prompt queries join mixed decode cohorts." % (
                ep, pool // ep, pool, "from the slots a group frees per step and the steps one import spans at Table I write rates, x4 margin"
                if import_pool is None else "as the smallest refill pool keeping every admission one step beyond the measured native import span upper bound (user decision 2026-10-24)"),
            *(("Attention data parallelism over %d EP groups: each group owns B/ep requests and runs their attention, dense MLP, router and shared expert on its own TP ranks (dense weights replicated per group, KV only on the owner group); MoE experts are shared by native all_to_all dispatch and combine across groups, and every layer ends at one cross-group barrier." % ep,) if ep > 1 else ()),
            *(("Pipeline scheduling uses %d request microbatch(es) per EP group, FIFO at each stage and a flush at every cohort boundary; request batching, Prefill chunks and token ubatches retain their own limits." % pipeline_microbatches,)
              if pp > 1 or pipeline_microbatches > 1 else ()),
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
    # The heuristic placement is deterministic; its time limit only aborts
    # candidate generation. 405B TP8 takes ~16 s alone, so the 60 s default
    # failed candidates spuriously when scan workers shared the CPU.
    options = {"design_decode_batch_size": batch_size, "design_prefill_tokens": 1,
        "operator_targets": operator_targets,
        "kv_cache_target": weight0, "time_limit_s": CONTROL_PLANE_TIME_LIMIT_S}
    return replace(s, placement=replace(s.placement,
        metadata={**s.placement.metadata, "control_plane": {"policy": {"options": options}}}))


PREFILL_NODE_ASSUMPTION = (
    "TTFT includes Prefill: an HBM4 Prefill replica with the Decode replica's tp/pp/ep layout (user approved "
    "2026-10-25: when HBM4 cannot hold the weights at the Decode tp, the smallest larger power-of-two tp that "
    "holds them within one 8-GPU node, same pp/ep; the candidate's prefill_pass records it) prefills each "
    "request alone the moment it arrives (no queueing) over its complete first prompt-1 tokens, without "
    "additional scheduler chunk limits while retaining model-required chunks. The pass ends when the last "
    "layer of its final chunk completes, where the KV export to the Decode node starts. Multi-cohort "
    "passes run entirely natively and directly observe final-layer completion, excluding the output head; "
    "single-cohort fast passes retain layer sampling. The "
    "Decode node computes the final prompt token and the first output token, so TTFT = Prefill pass + "
    "ingress TTFT (KV import start to first token).")


def build_native_prefill_scenario(model_id: str, workload_id: str, *, tp: int = 8, pp: int = 1, ep: int = 1,
                                  full_pass: bool = False, model_override=None, prompt_override=None):
    """The Prefill replica of a native deployment, running one request's pass.

    Same model, tp/pp/ep layout and per-rank structure as the Decode replica,
    on HBM4 for every memory configuration, without the PD source: a single
    request of prompt-1 tokens and one output. Scheduler budgets cover the
    complete prompt; model-required chunk boundaries remain in effect. A
    multi-cohort pass runs entirely natively and directly observes its final
    layer's completion, excluding the output head. A single-cohort fast pass
    retains layer sampling.
    """
    from heterollm_sim.simulation_options import NativePrefillPassConfig

    decode = build_native_comparison_scenario(model_id, "HBM4", workload_id, batch_size=ep, tp=tp, pp=pp, ep=ep,
        model_override=model_override, prompt_override=prompt_override, output_override=2)
    workload = next(row for row in workload_catalog() if row.workload_id == workload_id)
    tokens = (workload.input_tokens if prompt_override is None else prompt_override) - 1
    hardware = replace(decode.hardware, name=decode.hardware.name.replace("native-paper-", "native-prefill-"),
        components=tuple(c for c in decode.hardware.components if c.component_id != "pd_host"),
        links=tuple(link for link in decode.hardware.links if "pd_host" not in {
            link.source_component, link.target_component}))
    scheduler = replace(decode.workload.scheduler, max_num_seqs=ep, max_num_batched_tokens=tokens,
        max_num_ubatch_tokens=tokens, prefill_chunk_tokens=tokens, slo_tpot_ns=None)
    control = decode.placement.metadata["control_plane"]
    options = {**control["policy"]["options"], "design_prefill_tokens": tokens}
    kept = [value for value in decode.assumptions if value.startswith(("Native serving,", "Paper Table I",
        "Native generic NAND", "8x40MB", "Attention data parallelism"))]
    return replace(decode, name=decode.name.replace("native-", "native-prefill-", 1).rsplit("-B", 1)[0],
        hardware=hardware,
        placement=replace(decode.placement, hardware_name=hardware.name,
            metadata={**decode.placement.metadata, "control_plane": {**control, "policy": {
                **control["policy"], "options": options}}}),
        workload=replace(decode.workload, name="native-prefill-node",
            requests=(RequestSpec("native-prefill-request", 0., tokens, 1),), request_count=1,
            scheduler=scheduler, metadata={"native_paper_comparison": True}),
        simulation=SimulationConfig(mode="fast", native_prefill=NativePrefillPassConfig(full_pass=full_pass)),
        assumptions=(*kept, PREFILL_NODE_ASSUMPTION))
