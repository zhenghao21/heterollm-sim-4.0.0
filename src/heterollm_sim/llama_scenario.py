"""Apply a llama.cpp runtime contract to an authored scenario.

This adapter is deliberately small: it projects llama.cpp's scheduler/KV and
offload settings onto the existing V4 scheduler, KV policy, and control-plane
placement contracts, then runs the normal placement planner so the resulting
scenario contains real CPU/GPU ownership evidence.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .config import ScenarioConfig
from .control_plane_planner import PlacementPolicy, plan_runtime_placement
from .ir import KVCachePolicy, PlacementSpec, SchedulerSpec
from .runtime_adapters import LlamaCppRuntimeConfig


def apply_llama_runtime_config(
    scenario: ScenarioConfig,
    config: LlamaCppRuntimeConfig,
    *,
    materialize_placement: bool = True,
) -> ScenarioConfig:
    """Return ``scenario`` with llama.cpp semantics lowered into typed fields."""
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    if not isinstance(config, LlamaCppRuntimeConfig):
        raise TypeError("config must be a LlamaCppRuntimeConfig")
    if config.kv_type_k is not None and config.kv_type_v is not None and config.kv_type_k.casefold() != config.kv_type_v.casefold():
        raise ValueError("simulator currently requires identical llama.cpp K/V cache dtypes")
    for request in scenario.workload.requests:
        required = int(request.prompt_tokens) + int(request.output_tokens)
        if required > config.context:
            raise ValueError(
                "llama.cpp context {} is smaller than request {} token span {}"
                .format(config.context, request.request_id, required)
            )
    if not scenario.workload.requests:
        required = int(scenario.workload.prompt_tokens) + int(scenario.workload.output_tokens)
        if required > config.context:
            raise ValueError(
                "llama.cpp context {} is smaller than synthetic request token span {}"
                .format(config.context, required)
            )
    scheduler = scenario.workload.scheduler
    scheduler = replace(
        scheduler,
        max_num_seqs=config.parallel,
        max_num_batched_tokens=config.batch,
        max_num_ubatch_tokens=config.ubatch,
        mode="continuous" if config.cont_batching else "static",
    )
    workload = replace(
        scenario.workload,
        scheduler=scheduler,
        metadata={
            **scenario.workload.metadata,
            "llama_cpp_runtime": config.to_dict(),
            "context_limit_semantics": "per_slot_runtime_limit",
        },
    )
    kv = scenario.placement.kv_policy
    # llama.cpp's context is a per-slot reservation; preserve explicit KV
    # placement/offload targets while changing the slot geometry and dtype.
    kv_cache_component = kv.cache_component
    if not config.offload_kqv:
        host_memory = next(
            (component.component_id for component in scenario.hardware.components
             if component.normalized_kind in {"host_memory", "dram", "ddr", "ddr_memory"}),
            None,
        )
        if host_memory is None:
            raise ValueError("llama.cpp --no-kv-offload requires a host memory component")
        kv_cache_component = host_memory
    kv = replace(
        kv,
        cache_component=kv_cache_component,
        tokens_per_page=max(1, kv.tokens_per_page),
        dtype=config.kv_type_k or config.kv_type_v or kv.dtype,
    )
    placement_metadata = dict(scenario.placement.metadata)
    control = dict(placement_metadata.get("control_plane", {}))
    policy = dict(control.get("policy", {}))
    options = dict(policy.get("options", {}))
    options.update({
        # This llama.cpp build counts the output layer in -ngl: -ngl=12
        # offloads the output layer plus the last 11 repeating blocks.
        "gpu_loadable_layers": scenario.model.num_layers + 1 if config.gpu_layers < 0 else config.gpu_layers,
        "gpu_loadable_order": "tail",
        # llama.cpp keeps token_embd.weight CPU-resident when the output
        # matrix is a logical tied alias, then materializes a GPU runtime copy
        # for lm_head.  This avoids pulling the full embedding tensor over
        # PCIe on every decode token.
        "tied_weight_runtime_copies": config.gpu_layers != 0 and scenario.model.output_weight_bytes <= 0,
    })
    policy["options"] = options
    control["policy"] = policy
    evidence = dict(control.get("evidence", {}))
    evidence["llama_cpp_runtime"] = config.to_dict()
    evidence["llama_cpp_runtime_fingerprint"] = config.fingerprint
    control["evidence"] = evidence
    placement_metadata["control_plane"] = control
    placement_metadata["llama_cpp_runtime"] = config.to_dict()
    placement_metadata["llama_cpp_runtime_fingerprint"] = config.fingerprint
    placement = replace(scenario.placement, kv_policy=kv, metadata=placement_metadata)
    lowered = replace(
        scenario,
        placement=placement,
        workload=workload,
        llama_cpp_config=config,
        fusion_policy=replace(scenario.fusion_policy, flash_attention=config.flash_attn),
        assumptions=tuple(dict.fromkeys((*scenario.assumptions, "llama.cpp runtime config lowered into typed scheduler/KV/offload semantics"))),
    )
    if materialize_placement:
        decision = plan_runtime_placement(
            lowered,
            PlacementPolicy(
                gpu_loadable_layers=options["gpu_loadable_layers"],
                gpu_loadable_order=options["gpu_loadable_order"],
                tied_weight_runtime_copies=options["tied_weight_runtime_copies"],
            ),
        )
        lowered = decision.apply(lowered)
        # Keep the runtime evidence alongside planner evidence after apply().
        metadata = dict(lowered.placement.metadata)
        cp = dict(metadata.get("control_plane", {}))
        ev = dict(cp.get("evidence", {}))
        ev.update({"llama_cpp_runtime": config.to_dict(), "llama_cpp_runtime_fingerprint": config.fingerprint})
        cp["evidence"] = ev
        lowered = replace(lowered, placement=replace(lowered.placement, metadata={**metadata, "control_plane": cp}))
    return lowered


__all__ = ["apply_llama_runtime_config"]
