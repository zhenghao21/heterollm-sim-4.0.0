"""Analytical llama CUDA buffers: HBM first, directly accessed HBF overflow.

This is a deterministic extension of source loading rules, not evidence that
an unmodified native llama.cpp binary exposes HBF as a CUDA buffer type.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import ScenarioConfig, _host_memory_profile_from_dict
from .control_plane_planner import (
    PlacementPolicy, _base_capacity_usage, _component_capacities,
    _derive_requirements, _load_policy_targets, _mapping_controls,
    _split_tied_weight_runtime_copies,
)
from .ir import model_graph_execution_view
from .parallel import build_parallel_plan
from .serde import stable_hash


def device_memory_input_fingerprint(scenario: ScenarioConfig) -> str:
    """Exclude generated placement/evidence to avoid self-referential hashes."""
    return stable_hash({
        "hardware": scenario.hardware, "model": scenario.model,
        "profiles": scenario.component_profiles, "parallel": scenario.placement.parallel,
        "runtime": scenario.llama_cpp_config.to_dict(),
        "kv_dtype": scenario.placement.kv_policy.dtype,
        "kv_page_tokens": scenario.placement.kv_policy.tokens_per_page,
        "weights_resident": scenario.weights_resident,
    })


def _device_buffers(scenario: ScenarioConfig) -> tuple[ScenarioConfig, dict[str, Any]]:
    """Associate physical buffers with their directly attached CUDA device."""
    gpu_ids = {c.component_id for c in scenario.hardware.components
               if c.normalized_kind in {"gpu", "cuda"}}
    adjacency: dict[str, set[str]] = {}
    for link in scenario.hardware.links:
        adjacency.setdefault(link.source_component, set()).add(link.target_component)
        adjacency.setdefault(link.target_component, set()).add(link.source_component)
    profiles = {kind: dict(values) for kind, values in scenario.component_profiles.items()}
    components, buffers = [], {}
    for component in scenario.hardware.components:
        kind = component.normalized_kind
        if kind not in {"hbm", "gddr", "gddr_memory", "hbf"}:
            components.append(component)
            continue
        candidates = adjacency.get(component.component_id, set()) & gpu_ids
        declared = component.metadata.get("device_id")
        if declared is not None:
            candidates &= {declared}
        if len(candidates) != 1:
            raise ValueError("llama.cpp device memory {} requires one directly attached GPU".format(component.component_id))
        device = next(iter(candidates))
        if kind == "hbf":
            template = component.metadata.get("cost_profile_template")
            profile_id = component.cost_profile_id
            existing = [registry[profile_id] for key, registry in profiles.items()
                        if key in {"host_memory", "hbm"} and profile_id in registry]
            if len(existing) == 1:
                profile = existing[0]
            elif isinstance(template, dict):
                profile_id = profile_id or component.component_id + ".llama_memory"
                profile = _host_memory_profile_from_dict(template)
                profiles.setdefault("host_memory", {})[profile_id] = profile
            else:
                raise ValueError("HBF {} requires a configured memory cost profile or cost_profile_template".format(component.component_id))
            # Do not substitute a read ceiling for unknown program throughput.
            if (component.write_bandwidth_gbps <= 0
                    and getattr(profile, "write_bandwidth_gb_s", None) is None):
                raise ValueError("HBF {} requires explicit write bandwidth".format(component.component_id))
            read = component.read_bandwidth_gbps or float(profile.effective_read_bandwidth_gb_s) * 8
            write = component.write_bandwidth_gbps or float(profile.effective_write_bandwidth_gb_s) * 8
            component = replace(component, cost_profile_id=profile_id,
                read_bandwidth_gbps=read, write_bandwidth_gbps=write,
                bandwidth_gbps=component.bandwidth_gbps or max(read, write),
                metadata={**component.metadata, "access_mode": "memory", "read_only": False,
                    "writable": True, "write_buffer_bytes": 0, "bandwidth_mode": "directional",
                    "memory_service_owner": profile.resource_id,
                    "read_latency_ns": profile.read_latency_ns, "write_latency_ns": profile.write_latency_ns,
                    "transfer_granularity_bytes": profile.transaction_bytes,
                    "max_outstanding_requests": profile.max_outstanding_requests})
        buffers[component.component_id] = {
            "device_id": device, "buffer_kind": "HBF" if kind == "hbf" else "HBM",
            "access": "direct", "priority": 1 if kind == "hbf" else 0,
        }
        components.append(component)
    return replace(scenario, hardware=replace(scenario.hardware, components=tuple(components)),
                   component_profiles=profiles), buffers


def prepare_device_memory_policy(
    scenario: ScenarioConfig, policy: PlacementPolicy,
) -> tuple[ScenarioConfig, PlacementPolicy]:
    """Reserve static state, then allocate weights against the same byte ledger.

    The existing placement planner remains responsible for route validation,
    physical tensor aliases, generated placement, and final capacity checks.
    """
    from .serving import _kv_bytes_for_layer, _linear_state_bytes_per_layer

    config = scenario.llama_cpp_config
    scenario, buffers = _device_buffers(scenario)
    view = model_graph_execution_view(scenario.model.graph, schema_version=scenario.model.schema_version)
    parallel = build_parallel_plan(scenario, view)
    if parallel.tp_degree != 1 or parallel.ep_degree != 1:
        raise ValueError("llama.cpp HBM/HBF device-memory tiering currently requires TP=EP=1")
    components = scenario.hardware.component_map()
    host = next((c.component_id for c in scenario.hardware.components
                 if c.normalized_kind in {"host_memory", "dram", "ddr", "ddr_memory"}
                 and c.is_active_memory and c.is_writable), None)
    cpus = [c.component_id for c in scenario.hardware.components if c.normalized_kind == "cpu"]
    requirements = _derive_requirements(scenario, view,
        split_tied_runtime_copies=_split_tied_weight_runtime_copies(scenario, policy))
    load_targets = _load_policy_targets(policy, requirements)
    generated = _mapping_controls(scenario, []).previous_generated_tensor_ids
    used = _base_capacity_usage(scenario, generated | {r.tensor_id for r in requirements})
    capacities = _component_capacities(scenario)
    state_bytes: dict[str, int] = {}
    weight_bytes: dict[str, int] = {}
    kv_targets: dict[str, str] = {}
    state_targets: dict[str, str] = {}
    weight_targets: dict[str, str] = {}
    operator_targets: dict[str, str] = {}
    layers = tuple(item.layer for item in view.layer_instances)
    gpu_start = max(0, len(layers) + 1 - int(policy.gpu_loadable_layers))
    layer_gpu = {layer.layer_id: index >= gpu_start for index, layer in enumerate(layers)}

    def gpu_for(layer: Any) -> str:
        return parallel.ranks_for_layer(layer)[0].component_id

    def allocate(name: str, size: int, device: str | None, ledger: dict[str, int]) -> str:
        choices = ([host] if device is None else sorted(
            (key for key, value in buffers.items() if value["device_id"] == device),
            key=lambda key: (buffers[key]["priority"], key)))
        for target in choices:
            if target is not None and used.get(target, 0) + size <= capacities.get(target, 0):
                used[target] = used.get(target, 0) + size
                ledger[target] = ledger.get(target, 0) + size
                return target
        raise ValueError("llama.cpp shared device-memory capacity exhausted for {} ({} bytes; device {})".format(name, size, device or "CPU"))

    page = scenario.placement.kv_policy.tokens_per_page
    reserved_tokens = ((config.context + page - 1) // page) * page
    if not config.kv_unified:
        reserved_tokens *= config.parallel
    for layer in layers:
        device = gpu_for(layer) if config.offload_kqv and layer_gpu[layer.layer_id] else None
        if layer.is_linear_attention:
            size = _linear_state_bytes_per_layer(layer) * config.parallel
            state_targets[layer.layer_id] = allocate(layer.layer_id + ".linear_state", size, device, state_bytes)
        else:
            size = _kv_bytes_for_layer(scenario, layer, scenario.placement.kv_policy.dtype)[1] * reserved_tokens
            kv_targets[layer.layer_id] = allocate(layer.layer_id + ".kv_cache", size, device, state_bytes)
    # Requirements retain source graph order. First-fit within each priority
    # avoids consuming HBF until no HBM bank can hold this physical tensor.
    for requirement in requirements:
        if requirement.state_tensor:
            continue
        target_kind = load_targets.get(requirement.item_id, "gpu")
        device = gpu_for(requirement.layer or layers[-1]) if target_kind == "gpu" else None
        if requirement.mapping_key:
            if device is None and not cpus:
                raise ValueError("llama.cpp host operators require a CPU component")
            operator_targets[requirement.mapping_key] = device or cpus[0]
        if requirement.tensor_id and requirement.tensor_bytes > 0:
            weight_targets[requirement.tensor_id] = allocate(
                requirement.tensor_id, requirement.tensor_bytes, device, weight_bytes)
    metadata = dict(scenario.placement.metadata)
    metadata["llama_backend_memory"] = buffers
    metadata["llama_cpp_kv_layer_components"] = kv_targets
    metadata["llama_cpp_linear_state_layer_components"] = state_targets
    metadata["llama_cpp_kv_contract"] = {
        **metadata.get("llama_cpp_kv_contract", {}), "kv_layer_components": kv_targets,
        "buffer_policy": "HBM_first_HBF_overflow_direct",
    }
    metadata["memory_tiers"] = {
        "kv_layer_components": kv_targets, "linear_state_layer_components": state_targets,
    }
    metadata["llama_device_memory_policy"] = {
        "schema": "llama.cpp.device-memory-policy/v1",
        "mode": "source_rule_approximation", "native_dispatch_proven": False,
        "accuracy_validated": False, "runtime_fingerprint": config.fingerprint,
        "allocation_order": "static_context_state_then_weights; HBM_before_HBF",
        "reserved_kv_tokens": reserved_tokens, "reserved_state_slots": config.parallel,
        "state_reserved_bytes": state_bytes, "weight_bytes": weight_bytes,
        "total_allocated_bytes": used, "capacity_bytes": capacities,
        "scope": "one TP/EP rank per pipeline stage; distinct directly accessed buffers on each CUDA device",
    }
    kv_default = next(iter(kv_targets.values()), None)
    state_default = next(iter(state_targets.values()), None)
    placement = replace(scenario.placement, metadata=metadata,
        kv_policy=replace(scenario.placement.kv_policy, cache_component=kv_default,
                          offload_component=None, layout_mode="llama_static_layer"))
    scenario = replace(scenario, placement=placement,
        assumptions=tuple(dict.fromkeys((*scenario.assumptions,
            "HBM/HBF analytical source-rule allocation; buffers share CUDA device ownership and direct access; native dispatch unverified"))))
    return scenario, replace(policy, operator_targets=operator_targets,
        weight_tensor_targets=weight_targets, kv_cache_target=kv_default,
        linear_state_target=state_default, kv_layer_targets=kv_targets,
        linear_state_layer_targets=state_targets)
