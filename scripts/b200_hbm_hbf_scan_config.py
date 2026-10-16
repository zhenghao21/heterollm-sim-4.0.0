"""Fixed B200 compute, explicit HBM3E versus strict OCP HBF scan inputs.

All timing results must come from the normal serving/physical runtime.  This
module only constructs inputs and never substitutes a throughput formula.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.architecture_presets import get_architecture_preset
from heterollm_sim.config import _component_profile_registries_from_dict, model_from_dict
from heterollm_sim.control_plane_planner import _derive_requirements
from heterollm_sim.ir import KVCachePolicy, ParallelSpec, RankMappingSpec, RequestSpec, model_graph_execution_layers
from heterollm_sim.model_presets import materialize_model_payload
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.schema_v4 import ControllerProfile, GPUControllerProfile

MODELS = ("qwen2_5-0_5b", "qwen2_5-1_5b", "qwen2_5-3b", "qwen2_5-7b")
MODEL_BILLIONS = dict(zip(MODELS, (0.5, 1.5, 3.0, 7.0)))
STRATEGIES = ("HBM", "HBF")
PREFILL_TOKENS = (64 * 1024, 256 * 1024, 512 * 1024)
DECODE_TOKENS = (1 * 1024, 5 * 1024, 10 * 1024)
HOST_STAGING_BYTES = 64 * 1024**2
PREFILL_CHUNK_TOKENS = 2048
HBM_CAPACITY_BYTES = 180_000_000_000
HBM_BANDWIDTH_GB_S = 8000.0
ARCHITECTURE_PRESET = "nvidia-b200-1gpu-2hbf-2hbm"


def _hardware_and_profiles(strategy):
    source = get_architecture_preset(ARCHITECTURE_PRESET).hardware
    memory_id = "hbm0" if strategy == "HBM" else "hbf0"
    physical = (resolve_physical_config("hbm3e_reference", overrides={
        "capacity_bytes": HBM_CAPACITY_BYTES, "stacks": 8,
        "dies_per_stack": 8, "die_density_gbits": 24,
        "interface_bandwidth_gb_s": HBM_BANDWIDTH_GB_S}) if strategy == "HBM"
        else resolve_physical_config("hbf_ocp_0_7"))
    read_gb_s = (physical.directional_bandwidth_gb_s("read") if strategy == "HBM"
                 else physical.host_read_bandwidth_gb_s)
    write_gb_s = (physical.directional_bandwidth_gb_s("write") if strategy == "HBM"
                  else physical.host_write_bandwidth_gb_s)
    shared_gb_s = (physical.physical_interface_bandwidth_gb_s if strategy == "HBM"
                   else physical.host_bandwidth_gb_s)
    owner = memory_id + (".hbm_fabric" if strategy == "HBM" else ".memory")
    memory = source.get_component(memory_id)
    profile = dict(memory.metadata["cost_profile_template"])
    profile.update(bandwidth_gb_s=shared_gb_s, read_bandwidth_gb_s=read_gb_s,
        write_bandwidth_gb_s=write_gb_s, resource_id=owner,
        name="B200 scan " + strategy,
        transaction_bytes=physical.burst_bytes if strategy == "HBM" else physical.host_granularity_bytes,
        read_latency_ns=physical.read_latency_ns if strategy == "HBM" else physical.page_read_ns,
        write_latency_ns=physical.write_latency_ns if strategy == "HBM" else physical.page_program_ns,
        max_outstanding_requests=physical.max_outstanding_requests,
        parallel_lanes=physical.lane_count if strategy == "HBM" else physical.parallel_units)
    metadata = {key: value for key, value in memory.metadata.items()
                if key not in {"memory_service", "physical_composition", "technology", "facts", "provenance"}}
    metadata.update(physical_memory_config=physical, cost_profile_template=profile,
        memory_service_owner=owner, transfer_granularity_bytes=profile["transaction_bytes"],
        read_latency_ns=profile["read_latency_ns"], write_latency_ns=profile["write_latency_ns"],
        max_outstanding_requests=profile["max_outstanding_requests"], parallel_lanes=profile["parallel_lanes"],
        scan_memory_contract={"strategy": strategy,
            "physical_preset": "hbm3e_reference" if strategy == "HBM" else "hbf_ocp_0_7",
            "organization": "8 x 24 GiB reference stacks (24 Gb x 8H), not vendor disclosed topology" if strategy == "HBM"
                else "64 equivalent array units and 16 host channels from strict physical preset"})
    if strategy == "HBF":
        metadata.update(access_mode="memory", writable=True, write_buffer_bytes=0,
            host_write_staging={"host_component_id": "hostmem0",
                                "capacity_bytes": HOST_STAGING_BYTES, "reserve_blocks": 1})
    memory = replace(memory, capacity_bytes=physical.capacity_bytes,
        bandwidth_gbps=shared_gb_s * 8, read_bandwidth_gbps=read_gb_s * 8,
        write_bandwidth_gbps=write_gb_s * 8,
        ports=tuple(replace(port, bandwidth_gbps=shared_gb_s * 8) for port in memory.ports), metadata=metadata)
    kept_ids = {"gpu0", "cpu0", "hostmem0", memory_id}
    links = []
    for link in source.links:
        if link.source_component not in kept_ids or link.target_component not in kept_ids:
            continue
        if memory_id in (link.source_component, link.target_component):
            link = replace(link, bandwidth_gbps=shared_gb_s * 8,
                # HBM links reference the media itself. HBF retains a separate
                # explicit package path, so array latency is not a link latency.
                latency_ns=0.0 if strategy == "HBM" else 100.0,
                metadata=({**link.metadata, "bandwidth_source": "memory_component",
                           "service_ref": memory_id + ".access"} if strategy == "HBM" else
                          {**link.metadata, "scan_link_basis": "explicit 100 ns analytical package hop"}))
        links.append(link)
    used_ports = {(link.source_component, link.source_port) for link in links} | {
        (link.target_component, link.target_port) for link in links}
    components = []
    for component in source.components:
        if component.component_id not in kept_ids:
            continue
        if component.component_id == memory_id:
            component = memory
        if component.component_id == "gpu0":
            component = replace(component, metadata={**component.metadata,
                "scan_active_memory": memory_id, "memory_total_capacity_gb": physical.capacity_bytes / 1e9,
                "memory_total_bandwidth_tb_per_s": shared_gb_s / 1000.0})
        component = replace(component, ports=tuple(
            replace(port, bandwidth_gbps=shared_gb_s * 8)
                if component.component_id == "gpu0" and port.port_id == memory_id else port
            for port in component.ports if (component.component_id, port.port_id) in used_ports))
        components.append(component)
    raw_profiles = {}
    for component in components:
        kind = "host_memory" if component.kind == "hbf" else component.kind
        parameters = dict(component.metadata["cost_profile_template"])
        if kind in {"hbm", "host_memory"}:
            parameters.pop("name", None)
        raw_profiles.setdefault(kind, {})[component.cost_profile_id] = parameters
    hardware = replace(source, name="B200 all-" + strategy, components=tuple(components), links=tuple(links),
        metadata={"source_architecture_preset": ARCHITECTURE_PRESET,
                  "scan_strategy": strategy,
                  "memory_endpoint_contract": "one logical memory endpoint; only selected medium is installed"})
    return hardware, _component_profile_registries_from_dict(raw_profiles), memory_id


def build_scan_scenario(model_id, architecture, prompt_tokens, output_tokens,
                        prefill_chunk_tokens=PREFILL_CHUNK_TOKENS):
    strategy = architecture
    if strategy not in STRATEGIES:
        raise ValueError("unknown scan strategy: " + str(strategy))
    if model_id not in MODELS:
        raise ValueError("model is outside the authorized 0.5B–7B scan")
    if min(prompt_tokens, output_tokens, prefill_chunk_tokens) <= 0:
        raise ValueError("token counts must be positive")
    base = build_reference_scenario()
    hardware, profiles, memory_id = _hardware_and_profiles(strategy)
    payload = materialize_model_payload(model_id)
    graph = payload["graph"]
    native = int(graph["attributes"]["max_sequence_length"])
    requested = prompt_tokens + output_tokens
    if requested > native:
        graph["attributes"]["max_sequence_length"] = requested
        for operator in graph["operators"]:
            if operator["op_kind"] == "model_input":
                operator["parameters"]["max_sequence_length"] = requested
        payload.setdefault("metadata", {})["study_context_extrapolation"] = {
            "native_max_tokens": native, "requested_tokens": requested,
            "accuracy_claim": False}
    model = model_from_dict(payload)
    layers = model_graph_execution_layers(model.graph)
    chunk = min(prefill_chunk_tokens, prompt_tokens)
    scheduler = replace(base.workload.scheduler, max_num_seqs=1,
        max_num_batched_tokens=chunk, max_num_ubatch_tokens=chunk,
        prefill_chunk_tokens=chunk, preemption_enabled=False,
        slo_ttft_ns=None, slo_tbt_ns=None)
    parallel = ParallelSpec(rank_mapping=(RankMappingSpec(0, "gpu0", 0, 0, 0,
        memory_component_id=memory_id),), layer_to_stage={layer.layer_id: 0 for layer in layers})
    scenario = replace(base, name=f"b200-{strategy}-{model_id}-p{prompt_tokens}-d{output_tokens}",
        hardware=hardware, component_profiles=profiles, model=model, cim_interconnect=None,
        runtime_profile=ControllerProfile(gpu_controllers={"gpu0": GPUControllerProfile()}),
        placement=replace(base.placement, hardware_name=hardware.name, model_name=model.name,
            parallel=parallel, tensor_to_component={}, tensor_bytes={}, op_to_component={}, metadata={},
            kv_policy=KVCachePolicy(cache_component=memory_id, dtype="bf16", tokens_per_page=16,
                allocation_policy="eager", preemption_mode="recompute", layout_mode="legacy_single")),
        workload=replace(base.workload, name="single-request-input-output-length-scan",
            requests=(RequestSpec("scan-request", 0., prompt_tokens, output_tokens),), request_count=1,
            scheduler=scheduler, mtp=None), weights_resident=True,
        assumptions=("K=1024 tokens; one request; BF16 weights and KV; no speculative decoding.",
            "B200 compute/cache preset retained; all model operators explicitly execute on gpu0.",
            "Weights are resident before TTFT timing; CPU and 32 decimal GB HOST DRAM remain installed.",
            "Full HBF removes every HBM component; HOST DRAM staging budget is 64 MiB.",
            "Longer-than-native contexts extrapolate performance only, with no model-quality claim.",
            "HBM3E and HBF timing/geometry are explicit analytical presets, not measured vendor devices.",
            "Unfused generic RoPE generates and stores broadcast-expanded sin/cos tables per invocation in both strategies.",
            f"Prefill chunk limit is {prefill_chunk_tokens} tokens; output is simulated token by token."))
    requirements = _derive_requirements(scenario)
    options = {"operator_targets": {item.mapping_key: "gpu0" for item in requirements
            if item.mapping_key and not item.state_tensor},
        "weight_tensor_targets": {item.tensor_id: memory_id for item in requirements
            if item.tensor_id and item.tensor_bytes > 0 and not item.state_tensor},
        "kv_cache_target": memory_id, "design_prefill_tokens": chunk, "design_decode_batch_size": 1}
    return replace(scenario, placement=replace(scenario.placement,
        metadata={"control_plane": {"policy": {"options": options}}}))


def describe_scan_parameters():
    hbm = _hardware_and_profiles("HBM")[0].get_component("hbm0").metadata["physical_memory_config"]
    hbf = resolve_physical_config("hbf_ocp_0_7")
    return {"models": list(MODELS), "model_billions": MODEL_BILLIONS,
        "strategies": list(STRATEGIES), "prompt_tokens": list(PREFILL_TOKENS),
        "output_tokens": list(DECODE_TOKENS), "point_count": 72,
        "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS, "request_count": 1,
        "batch_size": 1, "gpu_preset": "nvidia-b200-sxm-gpu",
        "weights_dtype": "bf16", "kv_dtype": "bf16", "activation_dtype": "bf16",
        "weights_resident": True, "gpu_compute_peak_bf16_tflops": 2250.0,
        "rope_table_strategy": "per_invocation_broadcast_expanded_sin_cos; generation and backing writes included",
        "gpu_cache_l1_shared_bytes": 30 * 1024**2, "gpu_cache_l2_bytes": 50 * 1024**2,
        "hbm_capacity_bytes": HBM_CAPACITY_BYTES, "hbm_interface_bandwidth_gb_s": HBM_BANDWIDTH_GB_S,
        "hbm_physical_preset": "hbm3e_reference", "hbf_physical_preset": "hbf_ocp_0_7",
        "hbm_unconfigured_timing_parameters": hbm.metadata.get("unconfigured_timing_parameters", []),
        "hbm_timing_scope": "standard reference; timing packs only support 32Gb x 16H and are not applied to this 24Gb x 8H organization",
        "hbm_geometry": {"stacks": 8, "die_density_gbits": 24, "dies_per_stack": 8,
                          "raw_capacity_bytes": 192 * 1024**3},
        "hbf_parameters": {"capacity_bytes": hbf.capacity_bytes, "array_units": hbf.parallel_units,
            "host_channels": hbf.host_channels, "read_bandwidth_gb_s": hbf.host_read_bandwidth_gb_s,
            "write_bandwidth_gb_s": hbf.host_write_bandwidth_gb_s, "page_bytes": hbf.page_bytes,
            "read_granularity_bytes": hbf.host_granularity_bytes,
            "page_read_ns": hbf.page_read_ns, "page_program_ns": hbf.page_program_ns,
            "block_erase_ns": hbf.block_erase_ns, "gc_reserve_blocks": 1,
            "package_link_bandwidth_gb_s": 1600.0, "package_link_latency_ns": 100.0,
            "timing_basis": "explicit analytical preset assumptions; not OCP mandated device latencies"},
        "host_parameters": {"cpu_cores": 72, "cpu_frequency_ghz": 3.0,
            "dram_capacity_bytes": 32_000_000_000, "dram_bandwidth_gb_s": 44.8,
            "dram_latency_ns": 100.0, "pcie_generation": "5.0", "pcie_lanes": 16,
            "pcie_bandwidth_gb_s": 63.01538461538462, "pcie_link_latency_ns": 800.0},
        "host_staging_capacity_bytes": HOST_STAGING_BYTES,
        "host_hardware_source": ARCHITECTURE_PRESET,
        "ttft_definition": "arrival to first committed output token",
        "tpot_definition": "(last output token time - first output token time) / (output_tokens - 1)"}
