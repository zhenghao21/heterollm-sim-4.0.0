"""JSON configuration loading for complete simulation scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type, Union

from .cost_models import (
    CPUPipelineProfile,
    CPUProfile,
    CPUQuantizedDotCapability,
    CacheHierarchyProfile,
    CacheLevelProfile,
    DigitalSramCimProfile,
    GPUProfile,
    GPUQuantizedMatmulCapability,
    HBMProfile,
    HostGemmOffloadCapability,
    HostRecurrentOffloadCapability,
    HostMemoryProfile,
    HostOrchestrationProfile,
    TensorCoreProfile,
)
from .ir import (
    SCHEMA_VERSION,
    ComponentSpec,
    HardwareSpec,
    KVCachePolicy,
    LinkSpec,
    MTPPolicy,
    ModelSpec,
    ParallelSpec,
    PlacementSpec,
    PortSpec,
    RankMappingSpec,
    RequestSpec,
    SchedulerSpec,
    WorkloadSpec,
    default_memory_resource_id,
    normalize_component_kind,
)
from .serde import read_json
from .precision import dtype_bits
from .schema_v1 import model_graph_from_dict
from .runtime_adapters import LlamaCppRuntimeConfig
from .schema_v4 import (
    CONTROL_PLANE_FINGERPRINT_SCHEMA,
    ControllerProfile,
    runtime_profile_from_dict,
)


ComponentCostProfile = Union[
    GPUProfile,
    HBMProfile,
    CPUProfile,
    HostMemoryProfile,
    DigitalSramCimProfile,
]

_COMPONENT_PROFILE_TYPES: Mapping[str, Type[Any]] = {
    "gpu": GPUProfile,
    "hbm": HBMProfile,
    "cpu": CPUProfile,
    "host_memory": HostMemoryProfile,
    "cim": DigitalSramCimProfile,
}

# A hardware input owns the physical graph plus every profile that describes
# its components and control fabric.  Model, workload, placement, mapping,
# sampling, and llama.cpp request controls remain scenario inputs.
_HARDWARE_INPUT_PROFILE_FIELDS = (
    "components",
    "host_orchestration",
    "fusion",
    "cim_interconnect",
    "runtime",
)


def normalize_cost_profile_kind(component_kind: str) -> Optional[str]:
    """Map a hardware component kind to its typed cost-profile registry."""

    normalized = component_kind.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "gpu":
        return "gpu"
    if normalized == "cpu":
        return "cpu"
    if normalized in {"hbm", "hbm_stack"}:
        return "hbm"
    if normalized in {
        "host_memory",
        "dram",
        "ddr",
        "ddr_memory",
        "cxl_memory",
    }:
        return "host_memory"
    if normalized == "cim" or "compute_in_memory" in normalized or "cim" in normalized:
        return "cim"
    return None


def _schema_version(value: Any, field_name: str) -> str:
    version = str(value)
    if version != SCHEMA_VERSION:
        raise ValueError(
            "{} must be exactly {}; got {}".format(
                field_name, SCHEMA_VERSION, version
            )
        )
    return version


@dataclass(frozen=True)
class InterconnectProfile:
    resource_id: str
    bandwidth_gb_s: float
    latency_ns: float
    energy_pj_per_byte: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id.strip():
            raise ValueError("interconnect resource_id must not be empty")
        if (
            isinstance(self.bandwidth_gb_s, bool)
            or not isinstance(self.bandwidth_gb_s, (int, float))
            or not math.isfinite(float(self.bandwidth_gb_s))
            or self.bandwidth_gb_s <= 0
        ):
            raise ValueError("interconnect bandwidth_gb_s must be positive")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in (self.latency_ns, self.energy_pj_per_byte)
        ):
            raise ValueError("interconnect latency and energy must be non-negative")

    def transfer_ns(self, byte_count: int) -> float:
        if byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        return self.latency_ns + byte_count / self.bandwidth_gb_s


@dataclass(frozen=True)
class FusionPolicy:
    """Audited GPU-internal fusion groups enabled by a V4 scenario."""

    qkv_rope: bool
    flash_attention: bool
    gemm_epilogue_activation: bool
    residual_norm: bool
    max_fused_working_set_bytes: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "qkv_rope",
            "flash_attention",
            "gemm_epilogue_activation",
            "residual_norm",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError("fusion flags must be boolean")
        if (
            isinstance(self.max_fused_working_set_bytes, bool)
            or not isinstance(self.max_fused_working_set_bytes, int)
            or self.max_fused_working_set_bytes < 0
        ):
            raise ValueError(
                "max_fused_working_set_bytes must be a non-negative integer"
            )


@dataclass(frozen=True)
class HostOutputContract:
    """Typed destination and representation of host-visible model outputs."""

    target_component_id: str
    vocabulary_size: int
    logits_dtype: str
    logits_element_bytes: int
    allocation_semantics: str = "unspecified"

    def __post_init__(self) -> None:
        if not isinstance(self.target_component_id, str) or not self.target_component_id.strip():
            raise ValueError("host output target_component_id must not be empty")
        if not isinstance(self.logits_dtype, str) or not self.logits_dtype.strip():
            raise ValueError("host output logits_dtype must not be empty")
        if (
            isinstance(self.vocabulary_size, bool)
            or not isinstance(self.vocabulary_size, int)
            or self.vocabulary_size <= 0
        ):
            raise ValueError(
                "host output vocabulary_size must be a positive integer"
            )
        if (
            isinstance(self.logits_element_bytes, bool)
            or not isinstance(self.logits_element_bytes, int)
            or self.logits_element_bytes <= 0
        ):
            raise ValueError(
                "host output logits_element_bytes must be a positive integer"
            )
        bits = dtype_bits(
            self.logits_dtype,
            unsupported_message=(
                "unsupported host output logits_dtype {}".format(
                    self.logits_dtype
                )
            ),
        )
        if bits != self.logits_element_bytes * 8:
            raise ValueError(
                "host output logits dtype/element byte declarations disagree"
            )
        if not isinstance(self.allocation_semantics, str) or not self.allocation_semantics.strip():
            raise ValueError("host output allocation_semantics must not be empty")


@dataclass(frozen=True)
class SamplingPolicy:
    """Typed sampling selection; unsupported algorithms remain explicit."""

    mode: str
    temperature: Optional[float] = None
    implementation: Optional[str] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    min_keep: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise ValueError("sampling mode must not be empty")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature < 0
        ):
            raise ValueError("sampling temperature must be finite and non-negative")
        if self.mode.strip().lower() == "greedy" and self.temperature not in (
            None,
            0,
            0.0,
        ):
            raise ValueError("greedy sampling temperature must be zero or omitted")
        if self.implementation is not None and (
            not isinstance(self.implementation, str)
            or not self.implementation.strip()
        ):
            raise ValueError("sampling implementation must be non-empty text")
        implementation = (
            None
            if self.implementation is None
            else self.implementation.strip().lower()
        )
        if implementation not in (None, "llama_cpp_cpu_chain"):
            raise ValueError(
                "unsupported sampling implementation {}".format(
                    self.implementation
                )
            )
        chain_values = (self.top_k, self.top_p, self.min_p, self.min_keep)
        if implementation is None and any(
            value is not None for value in chain_values
        ):
            raise ValueError(
                "sampling chain parameters require an implementation"
            )
        if implementation == "llama_cpp_cpu_chain" and any(
            value is None for value in chain_values
        ):
            raise ValueError(
                "llama_cpp_cpu_chain requires top_k, top_p, min_p, and min_keep"
            )
        for field_name in ("top_k", "min_keep"):
            value = getattr(self, field_name)
            # Native min_keep=0 disables the minimum-count constraint; it
            # does not request an empty candidate set. top_k stays positive.
            minimum = 0 if field_name == "min_keep" else 1
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    "sampling {} must be a {} integer".format(
                        field_name,
                        "non-negative" if minimum == 0 else "positive",
                    )
                )
        for field_name in ("top_p", "min_p"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(
                    "sampling {} must be finite and in [0, 1]".format(
                        field_name
                    )
                )
        if implementation == "llama_cpp_cpu_chain":
            if self.mode.strip().lower() != "greedy":
                raise ValueError(
                    "llama_cpp_cpu_chain currently requires greedy mode"
                )
            if self.top_k is not None and self.top_k > 128:
                raise ValueError(
                    "llama_cpp_cpu_chain currently requires top_k <= 128"
                )
            if (
                self.top_k is not None
                and self.min_keep is not None
                and self.min_keep > self.top_k
            ):
                raise ValueError("sampling min_keep must not exceed top_k")


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    hardware: HardwareSpec
    model: ModelSpec
    placement: PlacementSpec
    workload: WorkloadSpec
    component_profiles: Mapping[str, Mapping[str, ComponentCostProfile]]
    host_orchestration_profile: HostOrchestrationProfile
    fusion_policy: FusionPolicy
    cim_interconnect: Optional[InterconnectProfile]
    host_output_contract: Optional[HostOutputContract] = None
    sampling_policy: Optional[SamplingPolicy] = None
    runtime_profile: ControllerProfile = field(default_factory=ControllerProfile)
    weights_resident: bool = True
    schema_version: str = SCHEMA_VERSION
    assumptions: Tuple[str, ...] = ()
    # Kept at the end so existing positional ScenarioConfig construction keeps
    # its historical field order.
    llama_cpp_config: Optional[LlamaCppRuntimeConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name must not be empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "scenario schema_version must be exactly {}".format(
                    SCHEMA_VERSION
                )
            )
        if not isinstance(self.hardware, HardwareSpec):
            raise ValueError("hardware must be a HardwareSpec")
        if not isinstance(self.model, ModelSpec):
            raise ValueError("model must be a ModelSpec")
        if not isinstance(self.placement, PlacementSpec):
            raise ValueError("placement must be a PlacementSpec")
        if not isinstance(self.workload, WorkloadSpec):
            raise ValueError("workload must be a WorkloadSpec")
        if self.cim_interconnect is not None and not isinstance(self.cim_interconnect, InterconnectProfile):
            raise ValueError("cim_interconnect must be an InterconnectProfile or None")
        if not isinstance(self.component_profiles, Mapping):
            raise ValueError("component_profiles must be a kind/id registry")
        for raw_kind, registry in self.component_profiles.items():
            kind = str(raw_kind)
            if kind not in _COMPONENT_PROFILE_TYPES:
                raise ValueError("unknown component profile kind: {}".format(kind))
            if not isinstance(registry, Mapping):
                raise ValueError(
                    "component profile registry {} must be a mapping".format(kind)
                )
            expected_type = _COMPONENT_PROFILE_TYPES[kind]
            for profile_id, profile in registry.items():
                if not isinstance(profile_id, str) or not profile_id.strip():
                    raise ValueError("component profile id must not be empty")
                if not isinstance(profile, expected_type):
                    raise ValueError(
                        "component profile {}.{} must be a {}".format(
                            kind, profile_id, expected_type.__name__
                        )
                    )
        if not isinstance(
            self.host_orchestration_profile, HostOrchestrationProfile
        ):
            raise ValueError(
                "host_orchestration_profile must be a HostOrchestrationProfile"
            )
        if not isinstance(self.fusion_policy, FusionPolicy):
            raise ValueError("fusion_policy must be a FusionPolicy")
        if self.host_output_contract is not None and not isinstance(
            self.host_output_contract, HostOutputContract
        ):
            raise ValueError(
                "host_output_contract must be a HostOutputContract or None"
            )
        if self.sampling_policy is not None and not isinstance(
            self.sampling_policy, SamplingPolicy
        ):
            raise ValueError("sampling_policy must be a SamplingPolicy or None")
        if not isinstance(self.runtime_profile, ControllerProfile):
            raise ValueError("runtime_profile must be a ControllerProfile")
        if self.llama_cpp_config is not None and not isinstance(
            self.llama_cpp_config, LlamaCppRuntimeConfig
        ):
            raise ValueError("llama_cpp_config must be a LlamaCppRuntimeConfig or None")
        component_map = self.hardware.component_map()
        if self.host_output_contract is not None:
            output_component = component_map.get(
                self.host_output_contract.target_component_id
            )
            if output_component is None:
                raise ValueError(
                    "host output component {} is absent from hardware".format(
                        self.host_output_contract.target_component_id
                    )
                )
            if normalize_cost_profile_kind(output_component.normalized_kind) not in {
                "cpu",
                "host_memory",
            }:
                raise ValueError(
                    "host output target must be CPU-visible memory or a CPU"
                )
        for component in self.hardware.components:
            profile_kind = self.component_profile_kind(component)
            if profile_kind is None:
                if component.cost_profile_id is not None:
                    raise ValueError(
                        "component {} kind {} cannot bind a typed cost profile".format(
                            component.component_id, component.kind
                        )
                    )
                continue
            if component.cost_profile_id is None:
                raise ValueError(
                    "typed cost-bearing component {} ({}) requires cost_profile_id".format(
                        component.component_id, component.kind
                    )
                )
            self.resolve_component_profile(component)
        self._bind_memory_services()
        for component_id, expected_kind in (
            (
                self.host_orchestration_profile.cpu_component_id,
                "cpu",
            ),
            (
                self.host_orchestration_profile.gpu_component_id,
                "gpu",
            ),
        ):
            component = component_map.get(component_id)
            if component is None:
                raise ValueError(
                    "orchestration component {} is absent from hardware".format(
                        component_id
                    )
                )
            if str(component.kind).lower() != expected_kind:
                raise ValueError(
                    "orchestration component {} must be a {}".format(
                        component_id, expected_kind
                    )
                )
        if not isinstance(self.weights_resident, bool):
            raise ValueError("weights_resident must be boolean")
        if not isinstance(self.assumptions, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.assumptions
        ):
            raise ValueError("assumptions must be a tuple of non-empty strings")

    def component_profile_kind(self, component: ComponentSpec) -> Optional[str]:
        """Resolve HBF's opt-in interface without reclassifying Flash by kind."""

        if component.normalized_kind == "hbf" and component.is_active_memory:
            if component.cost_profile_id is None:
                raise ValueError("HBF memory requires an explicit cost_profile_id")
            matches = [
                kind for kind in ("host_memory", "hbm")
                if component.cost_profile_id in self.component_profiles.get(kind, {})
            ]
            if len(matches) != 1:
                raise ValueError(
                    "HBF memory cost_profile_id must identify exactly one "
                    "host_memory or hbm profile"
                )
            return matches[0]
        return normalize_cost_profile_kind(component.normalized_kind)

    def resolve_component_profile(
        self,
        component: Union[ComponentSpec, str],
        expected_type: Optional[Type[Any]] = None,
    ) -> ComponentCostProfile:
        """Resolve the one typed cost profile explicitly bound to a component."""

        if isinstance(component, str):
            component = self.hardware.get_component(component)
        if not isinstance(component, ComponentSpec):
            raise TypeError("component must be a ComponentSpec or component_id")
        profile_kind = self.component_profile_kind(component)
        if profile_kind is None:
            raise ValueError(
                "component {} kind {} has no typed cost-profile registry".format(
                    component.component_id, component.kind
                )
            )
        profile_id = component.cost_profile_id
        if profile_id is None:
            raise ValueError(
                "component {} requires cost_profile_id".format(component.component_id)
            )
        registry = self.component_profiles.get(profile_kind, {})
        if profile_id not in registry:
            raise ValueError(
                "component {} references unknown {} cost profile {}".format(
                    component.component_id, profile_kind, profile_id
                )
            )
        profile = registry[profile_id]
        required_type = _COMPONENT_PROFILE_TYPES[profile_kind]
        if not isinstance(profile, required_type):
            raise ValueError(
                "component {} profile {} must be a {}".format(
                    component.component_id, profile_id, required_type.__name__
                )
            )
        if component.is_active_memory and isinstance(profile, (HBMProfile, HostMemoryProfile)):
            # Component bandwidth is the physical ceiling.  A profile remains
            # the source of calibrated/effective throughput, except when the
            # input omitted its bandwidth and the component must seed it.
            # ``memory_bandwidth_scope=aggregate`` is an explicit contract:
            # all matching banks share one profile owner and their physical
            # link capacities form the cap for that profile.
            caps = self._memory_bandwidth_caps(component)
            if caps[0] <= 0.0:
                # Profile-only memory tiers are valid.  Their profile value is
                # the only known ceiling and is checked by the profile class.
                caps = (
                    float(profile.bandwidth_gb_s) * 8.0,
                    float(profile.bandwidth_gb_s) * 8.0,
                    float(profile.bandwidth_gb_s) * 8.0,
                )
            shared_cap, read_cap, write_cap = (value / 8.0 for value in caps)
            if profile.measured_effective_bandwidth_gb_s is not None:
                measured = float(profile.measured_effective_bandwidth_gb_s)
                limit = min(read_cap, write_cap) if component.metadata.get("bandwidth_mode") == "directional" else shared_cap
                if measured > limit * (1.0 + 1e-12):
                    raise ValueError(
                        "component {} measured effective bandwidth exceeds physical cap"
                        .format(component.component_id)
                    )
            effective_shared = float(profile.effective_bandwidth_gb_s)
            if effective_shared > shared_cap * (1.0 + 1e-12):
                raise ValueError(
                    "component {} profile bandwidth exceeds physical cap"
                    .format(component.component_id)
                )
            if component.metadata.get("bandwidth_mode") == "directional":
                for direction, limit in (("read", read_cap), ("write", write_cap)):
                    effective = float(getattr(profile, "effective_" + direction + "_bandwidth_gb_s"))
                    if effective > limit * (1.0 + 1e-12):
                        raise ValueError(
                            "component {} profile {} bandwidth exceeds physical cap"
                            .format(component.component_id, direction)
                        )
            profile_resource_id = profile.resource_id
            owner = component.metadata.get("memory_service_owner")
            hbf_contract = component.normalized_kind == "hbf" and component.is_active_memory
            if owner and profile_resource_id == "hbm.channel" and component.normalized_kind != "hbf":
                profile_resource_id = str(owner)
            updates = {} if hbf_contract else {"resource_id": str(owner) if owner else profile_resource_id}
            if "memory_service_model" in component.metadata and hasattr(profile, "service_model"):
                updates["service_model"] = str(component.metadata["memory_service_model"])
            for metadata_name, profile_name in (
                ("read_latency_ns", "read_latency_ns"),
                ("write_latency_ns", "write_latency_ns"),
                ("transfer_granularity_bytes", "transaction_bytes"),
                ("max_outstanding_requests", "max_outstanding_requests"),
            ):
                if metadata_name in component.metadata and not hbf_contract:
                    updates[profile_name] = component.metadata[metadata_name]
            if not caps[0] and profile.bandwidth_gb_s <= 0:
                updates["bandwidth_gb_s"] = shared_cap
            if updates.get("resource_id") == profile.resource_id:
                updates.pop("resource_id")
            if not all(getattr(profile, key) == value for key, value in updates.items()):
                profile = replace(profile, **updates)
        if component.normalized_kind == "hbf" and component.is_active_memory:
            for name in ("read_latency_ns", "write_latency_ns",
                         "transaction_bytes", "max_outstanding_requests"):
                value = getattr(profile, name, None)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(float(value)) or value <= 0):
                    raise ValueError(
                        "HBF memory profile requires explicit positive {}".format(name)
                    )
            if component.metadata.get("memory_service_owner") != profile.resource_id:
                raise ValueError("HBF memory_service_owner must equal profile.resource_id")
            for media_name, profile_name in (
                ("transfer_granularity_bytes", "transaction_bytes"),
                ("max_outstanding_requests", "max_outstanding_requests"),
                ("read_latency_ns", "read_latency_ns"),
                ("write_latency_ns", "write_latency_ns"),
            ):
                value = component.metadata.get(media_name)
                if isinstance(value, bool) or value != getattr(profile, profile_name):
                    raise ValueError("HBF metadata.{} must equal profile.{}".format(media_name, profile_name))
            for direction in ("read", "write"):
                media_cap = component.directional_bandwidth_gbps(direction)
                effective = getattr(profile, "effective_" + direction + "_bandwidth_gb_s",
                                    profile.effective_bandwidth_gb_s)
                if media_cap <= 0:
                    # A single shared total is valid for both directions.  A
                    # zero directional field is no longer interpreted as a
                    # free write path.
                    media_cap = component.shared_bandwidth_gbps
                if media_cap <= 0 or effective * 8 > media_cap * (1 + 1e-12):
                    raise ValueError(
                        "HBF memory profile bandwidth must not exceed either explicit "
                        "read/write media bandwidth (direction={})".format(direction)
                    )
        if expected_type is not None and not isinstance(profile, expected_type):
            raise ValueError(
                "component {} profile {} is {}, expected {}".format(
                    component.component_id,
                    profile_id,
                    type(profile).__name__,
                    expected_type.__name__,
                )
            )
        return profile

    def _bind_memory_services(self) -> None:
        """Resolve physical memory once for operator and explicit endpoint access."""
        components = []
        for component in self.hardware.components:
            if not component.is_active_memory:
                components.append(component)
                continue
            # Generic topology bridges may be memory-like for routing purposes
            # without owning a typed memory service.  Keep them as topology
            # nodes; typed HBM/HBF/host-memory components still resolve through
            # the strict profile contract above.
            if self.component_profile_kind(component) is None:
                components.append(component)
                continue
            profile = self.resolve_component_profile(component)
            if not isinstance(profile, (HBMProfile, HostMemoryProfile)):
                components.append(component)
                continue
            metadata = dict(component.metadata)
            # Resolved values belong only in memory_service. Copying profile
            # defaults into authoring metadata would mask later profile edits.
            caps = self._memory_bandwidth_caps(component)
            if caps[0] <= 0:
                caps = (profile.bandwidth_gb_s * 8,) * 3
            metadata["memory_service"] = {
                "service_id": component.component_id + ".access",
                "physical_owner": profile.resource_id,
                "bandwidth_gb_s": profile.effective_bandwidth_gb_s,
                "read_bandwidth_gb_s": profile.effective_read_bandwidth_gb_s,
                "write_bandwidth_gb_s": profile.effective_write_bandwidth_gb_s,
                # Explicit names keep the calibrated rate distinct from the
                # physical ceiling for consumers that do not know the legacy
                # profile field names.
                "effective_bandwidth_gb_s": profile.effective_bandwidth_gb_s,
                "effective_read_bandwidth_gb_s": profile.effective_read_bandwidth_gb_s,
                "effective_write_bandwidth_gb_s": profile.effective_write_bandwidth_gb_s,
                "physical_bandwidth_gb_s": caps[0] / 8,
                "physical_read_bandwidth_gb_s": caps[1] / 8,
                "physical_write_bandwidth_gb_s": caps[2] / 8,
                "read_latency_ns": profile.read_latency_ns,
                "write_latency_ns": profile.write_latency_ns,
                "transaction_bytes": profile.transaction_bytes,
                "max_outstanding_requests": profile.max_outstanding_requests,
                "energy_pj_per_byte": profile.energy_pj_per_byte,
                "latency_scope": "memory_service",
                "service_model": getattr(profile, "service_model", metadata.get("memory_service_model", "analytical")),
            }
            components.append(replace(component, metadata=metadata))
        component_map = {item.component_id: item for item in components}
        links = []
        for link in self.hardware.links:
            metadata = dict(link.metadata)
            reference = metadata.get("service_ref")
            legacy_reference = metadata.get("bandwidth_source") in {"component", "memory_component", "shared_component"}
            if not reference and legacy_reference:
                memories = [component_map[name] for name in (link.source_component, link.target_component)
                            if name in component_map and component_map[name].is_active_memory]
                if len(memories) != 1:
                    raise ValueError("hardware.links[{}].metadata.service_ref requires one memory endpoint".format(link.link_id))
                reference = memories[0].component_id + ".access"
            if reference:
                memory = component_map.get(str(reference).removesuffix(".access"))
                if memory is None or not memory.is_active_memory or reference != memory.component_id + ".access":
                    raise ValueError("hardware.links[{}].metadata.service_ref={} is not a memory service".format(link.link_id, reference))
                if memory.component_id not in (link.source_component, link.target_component):
                    raise ValueError("hardware.links[{}].metadata.service_ref must reference an endpoint".format(link.link_id))
                service = memory.metadata["memory_service"]
                for field, expected in (("bandwidth_gbps", memory.shared_bandwidth_gbps),
                                        ("latency_ns", service["read_latency_ns"])):
                    value = getattr(link, field)
                    if value and not math.isclose(value, expected, rel_tol=1e-12):
                        raise ValueError("hardware.links[{}].{}={} conflicts with hardware.components[{}].{}={}; declare bandwidth_source=link for independent hardware".format(
                            link.link_id, field, value, memory.component_id,
                            "bandwidth_gbps" if field == "bandwidth_gbps" else "metadata.read_latency_ns", expected))
                metadata["service_ref"] = reference
                metadata["bandwidth_source"] = "memory_component"
                metadata["bandwidth_resource_id"] = service["physical_owner"]
                link = replace(link, bandwidth_gbps=0.0, latency_ns=0.0, metadata=metadata)
            links.append(link)
        components = tuple(components)
        links = tuple(links)
        # Keep ScenarioConfig.replace() cheap and identity-stable when the
        # already-resolved service contract is unchanged. Several planners
        # intentionally use object identity to prove placement-only edits.
        if components != self.hardware.components or links != self.hardware.links:
            object.__setattr__(
                self,
                "hardware",
                replace(self.hardware, components=components, links=links),
            )

    def _memory_bandwidth_caps(
        self, component: ComponentSpec
    ) -> Tuple[float, float, float]:
        """Return shared/read/write physical caps in Gb/s.

        Banks are aggregated only when the author explicitly marks the memory
        components with the same ``memory_aggregate_owner``.  Independent
        memory components therefore retain independent physical ceilings.
        """

        metadata = component.metadata
        scope = str(metadata.get("memory_bandwidth_scope", "per_component")).strip().lower()
        if scope != "aggregate" and component.cost_profile_id is not None:
            # A copied component metadata mapping may omit the marker while
            # its profile group still has explicitly marked aggregate banks.
            # Preserve that group contract across such metadata edits.
            scope = "aggregate" if any(
                item.cost_profile_id == component.cost_profile_id
                and str(item.metadata.get("memory_bandwidth_scope", "")).strip().lower() == "aggregate"
                for item in self.hardware.components
            ) else scope
        if scope != "aggregate":
            shared = component.shared_bandwidth_gbps
            return (
                shared,
                component.directional_bandwidth_gbps("read") or shared,
                component.directional_bandwidth_gbps("write") or shared,
            )
        owner = str(metadata.get("memory_aggregate_owner", "")).strip()
        if not owner and component.cost_profile_id is not None:
            owners = {
                str(item.metadata.get("memory_aggregate_owner", "")).strip()
                for item in self.hardware.components
                if item.cost_profile_id == component.cost_profile_id
                and str(item.metadata.get("memory_aggregate_owner", "")).strip()
            }
            if len(owners) == 1:
                owner = next(iter(owners))
        if not owner:
            raise ValueError(
                "component {} aggregate memory requires memory_aggregate_owner"
                .format(component.component_id)
            )
        linked_to_owner = {
            link.target_component
            for link in self.hardware.links
            if link.source_component == owner
        } | {
            link.source_component
            for link in self.hardware.links
            if link.target_component == owner
        }
        members = [
            item for item in self.hardware.components
            if item.is_active_memory
            and (not linked_to_owner or item.component_id in linked_to_owner)
            and (
                (
                    str(item.metadata.get("memory_bandwidth_scope", "")).strip().lower() == "aggregate"
                    and str(item.metadata.get("memory_aggregate_owner", "")).strip() == owner
                )
                or (
                    item.cost_profile_id == component.cost_profile_id
                    and not item.metadata.get("memory_bandwidth_scope")
                )
            )
        ]
        if not members:
            members = [component]
        return (
            sum(item.shared_bandwidth_gbps for item in members),
            sum(item.directional_bandwidth_gbps("read") or item.shared_bandwidth_gbps for item in members),
            sum(item.directional_bandwidth_gbps("write") or item.shared_bandwidth_gbps for item in members),
        )

    def _unique_compatibility_profile(
        self, kind: str, *, optional: bool = False
    ) -> Optional[ComponentCostProfile]:
        registry = self.component_profiles.get(kind, {})
        if not registry and optional:
            return None
        if len(registry) != 1:
            raise ValueError(
                "legacy scenario.{}_profile access requires exactly one {} profile; found {}".format(
                    kind, kind, len(registry)
                )
            )
        return next(iter(registry.values()))

    @property
    def gpu_profile(self) -> GPUProfile:
        return self._unique_compatibility_profile("gpu")  # type: ignore[return-value]

    @property
    def hbm_profile(self) -> HBMProfile:
        return self._unique_compatibility_profile("hbm")  # type: ignore[return-value]

    @property
    def cpu_profile(self) -> CPUProfile:
        return self._unique_compatibility_profile("cpu")  # type: ignore[return-value]

    @property
    def host_memory_profile(self) -> HostMemoryProfile:
        return self._unique_compatibility_profile("host_memory")  # type: ignore[return-value]

    @property
    def cim_profile(self) -> Optional[DigitalSramCimProfile]:
        return self._unique_compatibility_profile("cim", optional=True)  # type: ignore[return-value]


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("{} must be an object".format(field_name))
    return value


def _array(value: Any, field_name: str) -> Tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("{} must be an array".format(field_name))
    return tuple(value)


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError("{} must be an integer".format(field_name))
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("{} must be an integer".format(field_name))
    if isinstance(value, float) and value != converted:
        raise ValueError("{} must be an integer".format(field_name))
    return converted


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be a number".format(field_name))
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("{} must be a number".format(field_name))
    if not math.isfinite(converted):
        raise ValueError("{} must be finite".format(field_name))
    return converted


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError("{} must be boolean".format(field_name))
    return value


def _optional_string(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _reject_unknown_fields(
    data: Mapping[str, Any], field_name: str, allowed: Tuple[str, ...]
) -> None:
    unknown = tuple(sorted(set(data).difference(allowed)))
    if unknown:
        raise ValueError(
            "V4 {} contains unknown fields: {}".format(
                field_name, ", ".join(unknown)
            )
        )


def _reject_dataclass_unknown_fields(
    data: Mapping[str, Any], field_name: str, target: Any
) -> None:
    _reject_unknown_fields(
        data,
        field_name,
        tuple(field.name for field in fields(target)),
    )


_TRAINING_FIELDS = {
    "backward",
    "backward_pass",
    "checkpoint",
    "checkpoint_interval",
    "data_loader",
    "data_parallel",
    "data_parallel_degree",
    "data_pipeline",
    "dataloader",
    "dataset",
    "ddp",
    "distributed_data_parallel",
    "dp_degree",
    "epochs",
    "fsdp",
    "gradient",
    "gradient_accumulation_steps",
    "gradients",
    "learning_rate",
    "loss",
    "optimizer",
    "pipeline_schedule",
    "train_steps",
    "training",
    "training_mode",
    "training_pipeline",
    "weight_update",
    "zero",
    "zero_stage",
}


def _reject_training_fields(data: Mapping[str, Any], field_name: str) -> None:
    forbidden = sorted(_TRAINING_FIELDS.intersection(data))
    if forbidden:
        raise ValueError(
            "{} contains inference-schema-forbidden training fields: {}".format(
                field_name, ", ".join(forbidden)
            )
        )


def _ports(
    items: Any,
    schema_version: str = SCHEMA_VERSION,
    *,
    component_id: str = "",
) -> Tuple[PortSpec, ...]:
    if items is None:
        return ()
    ports = []
    for item in _array(items, "component ports"):
        data = _mapping(item, "port")
        _reject_dataclass_unknown_fields(data, "hardware component port", PortSpec)
        port_schema_version = _schema_version(
            data.get("schema_version", schema_version), "port schema_version"
        )
        ports.append(
            PortSpec(
                port_id=str(data.get("port_id", "")),
                protocol=str(data.get("protocol", "")),
                role=str(data.get("role", "")),
                direction=str(data.get("direction", "bidirectional")),
                version=str(data.get("version", "1.0")),
                lanes=_integer(data.get("lanes", 1), "port lanes"),
                bandwidth_gbps=_number(data.get("bandwidth_gbps", 0.0), "port bandwidth_gbps"),
                max_links=_integer(data.get("max_links", 1), "port max_links"),
                payload=_optional_string(data.get("payload")),
                metadata=_mapping(data.get("metadata", {}), "port metadata"),
                schema_version=port_schema_version,
            )
        )
    return tuple(ports)


def hardware_from_dict(data: Mapping[str, Any]) -> HardwareSpec:
    _reject_training_fields(data, "hardware")
    _reject_dataclass_unknown_fields(data, "hardware", HardwareSpec)
    schema_version = _schema_version(
        data.get("schema_version", SCHEMA_VERSION), "hardware schema_version"
    )
    components = []
    for raw_component in _array(data.get("components", []), "components"):
        values = _mapping(raw_component, "component")
        _reject_dataclass_unknown_fields(
            values, "hardware component", ComponentSpec
        )
        component_id = str(values.get("component_id", ""))
        component_schema_version = _schema_version(
            values.get("schema_version", schema_version),
            "component schema_version",
        )
        parsed_ports = _ports(
            values.get("ports"),
            component_schema_version,
            component_id=component_id,
        )
        declared_bandwidth = _number(
            values.get("bandwidth_gbps", 0.0), "component bandwidth_gbps"
        )
        read_bandwidth = _number(
            values.get("read_bandwidth_gbps", 0.0), "component read_bandwidth_gbps"
        )
        write_bandwidth = _number(
            values.get("write_bandwidth_gbps", 0.0), "component write_bandwidth_gbps"
        )
        if declared_bandwidth <= 0 and normalize_component_kind(str(values.get("kind", ""))) in {
            "hbm", "hbm_stack", "dram", "ddr", "ddr_memory", "host_memory",
            "cxl_memory", "memory", "sram", "shared_memory",
        }:
            # Authoring owns the result after parsing.  This only fills a
            # missing total from an already declared endpoint capability so
            # old topology presets do not silently create a second profile
            # budget.
            declared_bandwidth = max(
                read_bandwidth,
                write_bandwidth,
                *(float(port.bandwidth_gbps) for port in parsed_ports),
                0.0,
            )
        metadata = _mapping(values.get("metadata", {}), "component metadata")
        if (
            metadata.get("bandwidth_mode") is None
            and read_bandwidth > 0.0
            and write_bandwidth > 0.0
            and read_bandwidth != write_bandwidth
        ):
            # A component with two unequal one-way declarations is
            # directional even when the parser also derives a shared total
            # for compatibility with older schemas.
            metadata = dict(metadata)
            metadata["bandwidth_mode"] = "directional"
        components.append(
            ComponentSpec(
                component_id=component_id,
                kind=str(values.get("kind", "")),
                cost_profile_id=_optional_string(values.get("cost_profile_id")),
                ports=parsed_ports,
                package_id=str(values.get("package_id", "")),
                die_id=str(values.get("die_id", "")),
                capacity_bytes=_integer(values.get("capacity_bytes", 0), "component capacity_bytes"),
                peak_ops_per_s=_number(values.get("peak_ops_per_s", 0.0), "component peak_ops_per_s"),
                read_bandwidth_gbps=read_bandwidth,
                write_bandwidth_gbps=write_bandwidth,
                metadata=metadata,
                schema_version=component_schema_version,
                bandwidth_gbps=declared_bandwidth,
            )
        )
    links = []
    for raw_link in _array(data.get("links", []), "links"):
        values = _mapping(raw_link, "link")
        _reject_dataclass_unknown_fields(values, "hardware link", LinkSpec)
        link_schema_version = _schema_version(
            values.get("schema_version", schema_version), "link schema_version"
        )
        links.append(
            LinkSpec(
                link_id=str(values.get("link_id", "")),
                source_component=str(values.get("source_component", "")),
                source_port=str(values.get("source_port", "")),
                target_component=str(values.get("target_component", "")),
                target_port=str(values.get("target_port", "")),
                protocol=str(values.get("protocol", "")),
                version=str(values.get("version", "1.0")),
                lanes=_integer(values.get("lanes", 1), "link lanes"),
                bandwidth_gbps=_number(values.get("bandwidth_gbps", 0.0), "link bandwidth_gbps"),
                latency_ns=_number(values.get("latency_ns", 0.0), "link latency_ns"),
                bidirectional=_boolean(values.get("bidirectional", True), "link bidirectional"),
                payload=_optional_string(values.get("payload")),
                metadata=_mapping(values.get("metadata", {}), "link metadata"),
                schema_version=link_schema_version,
            )
        )
    return HardwareSpec(
        name=str(data.get("name", "")),
        components=tuple(components),
        links=tuple(links),
        require_connected=_boolean(data.get("require_connected", True), "hardware require_connected"),
        metadata=_mapping(data.get("metadata", {}), "hardware metadata"),
        schema_version=schema_version,
    )


def model_from_dict(data: Mapping[str, Any]) -> ModelSpec:
    _reject_training_fields(data, "model")
    _reject_unknown_fields(
        data,
        "model",
        (
            "schema_version",
            "name",
            "graph",
            "text_backbone_only",
            "supported_modalities",
            "excluded_subgraphs",
            "metadata",
        ),
    )
    schema_version = _schema_version(
        data.get("schema_version", SCHEMA_VERSION), "model schema_version"
    )
    if "graph" not in data:
        raise ValueError("V4 model.graph is required")
    graph = model_graph_from_dict(_mapping(data["graph"], "model graph"))
    return ModelSpec(
        name=str(data.get("name", "")),
        graph=graph,
        text_backbone_only=_boolean(
            data.get("text_backbone_only", True), "text_backbone_only"
        ),
        supported_modalities=tuple(
            str(item)
            for item in _array(
                data.get("supported_modalities", ["text"]),
                "supported_modalities",
            )
        ),
        excluded_subgraphs=tuple(
            str(item)
            for item in _array(
                data.get("excluded_subgraphs", []), "excluded_subgraphs"
            )
        ),
        metadata=_mapping(data.get("metadata", {}), "model metadata"),
        schema_version=schema_version,
    )


def rank_mapping_from_dict(data: Mapping[str, Any]) -> RankMappingSpec:
    _reject_dataclass_unknown_fields(
        data, "placement.parallel rank mapping", RankMappingSpec
    )
    return RankMappingSpec(
        rank=_integer(data.get("rank", -1), "rank"),
        component_id=str(data.get("component_id", "")),
        tp_rank=_integer(data.get("tp_rank", -1), "tp_rank"),
        pp_rank=_integer(data.get("pp_rank", -1), "pp_rank"),
        ep_rank=_integer(data.get("ep_rank", -1), "ep_rank"),
        memory_component_id=_optional_string(data.get("memory_component_id")),
        cim_component_id=_optional_string(data.get("cim_component_id")),
    )


def parallel_from_dict(data: Mapping[str, Any]) -> ParallelSpec:
    _reject_training_fields(data, "parallel")
    _reject_unknown_fields(
        data,
        "placement.parallel",
        (
            "tp_degree",
            "pp_degree",
            "ep_degree",
            "rank_mapping",
            "layer_to_stage",
            "collective_algorithm",
            "routing_policy",
            "allow_padding",
        ),
    )
    rank_items = data.get("rank_mapping", [])
    rank_mapping = tuple(
        rank_mapping_from_dict(_mapping(item, "rank mapping"))
        for item in _array(rank_items, "rank_mapping")
    )
    layer_to_stage_raw = _mapping(data.get("layer_to_stage", {}), "layer_to_stage")
    return ParallelSpec(
        tp_degree=_integer(data.get("tp_degree", 1), "parallel tp_degree"),
        pp_degree=_integer(data.get("pp_degree", 1), "parallel pp_degree"),
        ep_degree=_integer(data.get("ep_degree", 1), "parallel ep_degree"),
        rank_mapping=rank_mapping,
        layer_to_stage={
            str(layer_id): _integer(stage, "layer_to_stage value")
            for layer_id, stage in layer_to_stage_raw.items()
        },
        collective_algorithm=str(data.get("collective_algorithm", "auto")),
        routing_policy=str(data.get("routing_policy", "lowest_latency")),
        allow_padding=_boolean(
            data.get("allow_padding", True),
            "parallel allow_padding",
        ),
    )


def kv_policy_from_dict(data: Mapping[str, Any]) -> KVCachePolicy:
    _reject_unknown_fields(
        data,
        "placement.kv_policy",
        (
            "cache_component",
            "offload_component",
            "tokens_per_page",
            "dtype",
            "offload_ratio",
            "allocation_policy",
            "preemption_mode",
            "prefetch_distance",
            "layout_mode",
            "kv_unified",
            "pool_components",
        ),
    )
    return KVCachePolicy(
        cache_component=_optional_string(data.get("cache_component")),
        offload_component=_optional_string(data.get("offload_component")),
        tokens_per_page=_integer(
            data.get("tokens_per_page", 16),
            "kv_policy tokens_per_page",
        ),
        dtype=_optional_string(data.get("dtype")),
        offload_ratio=_number(data.get("offload_ratio", 1.0), "kv_policy offload_ratio"),
        allocation_policy=str(data.get("allocation_policy", "lazy")),
        preemption_mode=str(data.get("preemption_mode", "auto")),
        prefetch_distance=_integer(
            data.get("prefetch_distance", 0), "kv_policy prefetch_distance"
        ),
        layout_mode=str(data.get("layout_mode", "legacy_single")),
        kv_unified=_boolean(data.get("kv_unified", True), "kv_policy kv_unified"),
        pool_components=tuple(
            str(item)
            for item in _array(data.get("pool_components", ()), "kv_policy pool_components")
        ),
    )


def placement_from_dict(data: Mapping[str, Any]) -> PlacementSpec:
    _reject_training_fields(data, "placement")
    _reject_unknown_fields(
        data,
        "placement",
        (
            "model_name",
            "hardware_name",
            "op_to_component",
            "tensor_to_component",
            "tensor_bytes",
            "parallel",
            "kv_policy",
            "metadata",
            "schema_version",
        ),
    )
    parallel_raw = data.get("parallel")
    if parallel_raw is None:
        raise ValueError("V4 placement.parallel is required")
    parallel = parallel_from_dict(_mapping(parallel_raw, "parallel"))

    kv_raw = data.get("kv_policy")
    if kv_raw is None:
        raise ValueError("V4 placement.kv_policy is required")
    kv_policy = kv_policy_from_dict(_mapping(kv_raw, "kv_policy"))

    metadata = _mapping(data.get("metadata", {}), "placement metadata")
    if metadata.get("linear_state_offload_mode", "mirror") not in ("mirror", "pressure"):
        raise ValueError("linear_state_offload_mode must be mirror or pressure")
    if "auto_mapping" in metadata:
        raise ValueError(
            "V4 placement.metadata.auto_mapping is retired; run the explicit V3-to-V4 importer"
        )
    control_plane_raw = metadata.get("control_plane")
    if control_plane_raw is not None:
        control_plane = _mapping(
            control_plane_raw, "placement.metadata.control_plane"
        )
        _reject_unknown_fields(
            control_plane,
            "placement.metadata.control_plane",
            ("policy", "decision", "evidence"),
        )
        for section_name in ("policy", "decision", "evidence"):
            if section_name in control_plane:
                _mapping(
                    control_plane[section_name],
                    "placement.metadata.control_plane.{}".format(section_name),
                )
        policy = _mapping(
            control_plane.get("policy", {}),
            "placement.metadata.control_plane.policy",
        )
        retired_lock_fields = sorted(
            field_name
            for field_name in ("locked_op_keys", "locked_tensor_ids")
            if field_name in policy
        )
        if retired_lock_fields:
            raise ValueError(
                "V4 authoring cannot provide manual control-plane locks: {}".format(
                    ", ".join(
                        "placement.metadata.control_plane.policy.{}".format(field_name)
                        for field_name in retired_lock_fields
                    )
                )
            )
        _reject_unknown_fields(
            policy,
            "placement.metadata.control_plane.policy",
            ("options",),
        )
        options = _mapping(
            policy.get("options", {}),
            "placement.metadata.control_plane.policy.options",
        )
        _reject_unknown_fields(
            options,
            "placement.metadata.control_plane.policy.options",
            (
                "mode",
                "objective",
                "time_limit_s",
                "solver",
                "allow_cold_cim_streaming",
                "design_prefill_tokens",
                "design_decode_batch_size",
                "design_throughput_tokens",
                "gpu_loadable_layers",
                "gpu_loadable_order",
                "tied_weight_runtime_copies",
                "operator_targets",
                "weight_tensor_targets",
                "kv_cache_target",
                "linear_state_target",
                "linear_state_offload_target",
                "kv_layer_targets",
                "linear_state_layer_targets",
            ),
        )
        for name in ("operator_targets", "weight_tensor_targets", "kv_layer_targets", "linear_state_layer_targets"):
            if name in options:
                targets = _mapping(options[name], name)
                if any(not isinstance(key, str) or not key.strip()
                       or not isinstance(value, str) or not value.strip()
                       for key, value in targets.items()):
                    raise ValueError("{} must map non-empty names to component IDs".format(name))
        for name in ("kv_cache_target", "linear_state_target", "linear_state_offload_target"):
            if name in options and options[name] is not None:
                if not isinstance(options[name], str) or not options[name].strip():
                    raise ValueError("{} must be a non-empty component ID".format(name))
        if "tied_weight_runtime_copies" in options:
            _boolean(options["tied_weight_runtime_copies"], "tied_weight_runtime_copies")
        evidence = _mapping(
            control_plane.get("evidence", {}),
            "placement.metadata.control_plane.evidence",
        )
        fingerprint_schema = evidence.get("fingerprint_schema")
        if (
            fingerprint_schema is not None
            and fingerprint_schema != CONTROL_PLANE_FINGERPRINT_SCHEMA
        ):
            raise ValueError(
                "V4 control-plane fingerprint_schema must be exactly {}; got {}".format(
                    CONTROL_PLANE_FINGERPRINT_SCHEMA, fingerprint_schema
                )
            )

    authoring_placement = {
        "op_to_component": _mapping(
            data.get("op_to_component", {}), "op_to_component"
        ),
        "tensor_to_component": _mapping(
            data.get("tensor_to_component", {}), "tensor_to_component"
        ),
        "tensor_bytes": _mapping(data.get("tensor_bytes", {}), "tensor_bytes"),
    }
    populated_fields = sorted(
        field_name
        for field_name, values in authoring_placement.items()
        if values
    )
    if populated_fields:
        raise ValueError(
            "V4 authoring cannot provide manual placement fields: {}; runtime "
            "placement is materialized internally by the CPU control plane".format(
                ", ".join("placement.{}".format(name) for name in populated_fields)
            )
        )
    return PlacementSpec(
        model_name=str(data.get("model_name", "")),
        hardware_name=str(data.get("hardware_name", "")),
        op_to_component={},
        tensor_to_component={},
        tensor_bytes={},
        parallel=parallel,
        kv_policy=kv_policy,
        metadata=metadata,
        schema_version=_schema_version(
            data.get("schema_version", SCHEMA_VERSION),
            "placement schema_version",
        ),
    )


def scheduler_from_dict(data: Mapping[str, Any]) -> SchedulerSpec:
    _reject_unknown_fields(
        data,
        "workload.scheduler",
        (
            "mode",
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_num_ubatch_tokens",
            "prefill_chunk_tokens",
            "prefill_stop_offsets",
            "mixed_phase_batching",
            "policy",
            "phase_candidate_order",
            "starvation_ns",
            "preemption_enabled",
            "preemption_granularity",
            "preemption_policy",
            "slo_ttft_ns",
            "slo_tbt_ns",
        ),
    )
    policy = data.get("policy", "decode_first")
    return SchedulerSpec(
        mode=str(data.get("mode", "static")),
        max_num_seqs=_integer(data.get("max_num_seqs", 1), "scheduler max_num_seqs"),
        max_num_batched_tokens=_integer(
            data.get("max_num_batched_tokens", 2048),
            "scheduler max_num_batched_tokens",
        ),
        max_num_ubatch_tokens=(
            None
            if data.get("max_num_ubatch_tokens") is None
            else _integer(
                data.get("max_num_ubatch_tokens"),
                "scheduler max_num_ubatch_tokens",
            )
        ),
        prefill_chunk_tokens=_integer(
            data.get("prefill_chunk_tokens", 512),
            "scheduler prefill_chunk_tokens",
        ),
        prefill_stop_offsets=tuple(
            _integer(value, "scheduler prefill_stop_offsets value")
            for value in _array(data.get("prefill_stop_offsets", []), "scheduler prefill_stop_offsets")
        ),
        mixed_phase_batching=_boolean(
            data.get("mixed_phase_batching", False),
            "scheduler mixed_phase_batching",
        ),
        policy=str(policy),
        phase_candidate_order=str(
            data.get("phase_candidate_order", "least_recently_served")
        ),
        starvation_ns=_number(data.get("starvation_ns", 5_000_000), "scheduler starvation_ns"),
        preemption_enabled=_boolean(
            data.get("preemption_enabled", True), "scheduler preemption_enabled"
        ),
        preemption_granularity=str(data.get("preemption_granularity", "boundary")),
        preemption_policy=str(data.get("preemption_policy", "auto")),
        slo_ttft_ns=(
            None
            if data.get("slo_ttft_ns") is None
            else _number(data.get("slo_ttft_ns"), "scheduler slo_ttft_ns")
        ),
        slo_tbt_ns=(
            None
            if data.get("slo_tbt_ns") is None
            else _number(data.get("slo_tbt_ns"), "scheduler slo_tbt_ns")
        ),
    )


def mtp_from_dict(data: Mapping[str, Any]) -> MTPPolicy:
    _reject_unknown_fields(
        data,
        "workload.mtp",
        (
            "method",
            "candidate_tokens",
            "min_draft_tokens",
            "continuation_threshold",
            "proposal_length_model",
            "expected_draft_tokens_per_round",
            "draft_length_trace",
            "acceptance_model",
            "acceptance_rate",
            "proposal_cost_scale",
            "acceptance_trace",
        ),
    )
    trace_raw = data.get("acceptance_trace", [])
    draft_length_trace_raw = data.get("draft_length_trace", [])
    return MTPPolicy(
        method=str(data.get("method", "head_based")),
        candidate_tokens=_integer(
            data.get("candidate_tokens", 4),
            "mtp candidate_tokens",
        ),
        min_draft_tokens=_integer(
            data.get("min_draft_tokens", 0),
            "mtp min_draft_tokens",
        ),
        continuation_threshold=(
            None
            if data.get("continuation_threshold") is None
            else _number(
                data.get("continuation_threshold"),
                "mtp continuation_threshold",
            )
        ),
        proposal_length_model=str(
            data.get("proposal_length_model", "max")
        ),
        expected_draft_tokens_per_round=(
            None
            if data.get("expected_draft_tokens_per_round") is None
            else _number(
                data.get("expected_draft_tokens_per_round"),
                "mtp expected_draft_tokens_per_round",
            )
        ),
        draft_length_trace=tuple(
            _integer(value, "mtp draft_length_trace value")
            for value in _array(
                draft_length_trace_raw, "mtp draft_length_trace"
            )
        ),
        acceptance_model=str(data.get("acceptance_model", "expected")),
        acceptance_rate=(
            None
            if data.get("acceptance_rate") is None
            else _number(data.get("acceptance_rate"), "mtp acceptance_rate")
        ),
        proposal_cost_scale=_number(
            data.get("proposal_cost_scale", 0.15), "mtp proposal_cost_scale"
        ),
        acceptance_trace=tuple(
            _number(value, "mtp acceptance_trace value")
            for value in _array(trace_raw, "mtp acceptance_trace")
        ),
    )


def workload_from_dict(data: Mapping[str, Any]) -> WorkloadSpec:
    _reject_training_fields(data, "workload")
    _reject_unknown_fields(
        data,
        "workload",
        (
            "name",
            "requests",
            "request_count",
            "prompt_tokens",
            "output_tokens",
            "arrival_rate_rps",
            "random_seed",
            "scheduler",
            "mtp",
            "metadata",
            "schema_version",
        ),
    )
    schema_version = _schema_version(
        data.get("schema_version", SCHEMA_VERSION), "workload schema_version"
    )
    requests = []
    for raw_request in _array(data.get("requests", []), "requests"):
        values = _mapping(raw_request, "request")
        _reject_dataclass_unknown_fields(values, "workload request", RequestSpec)
        requests.append(
            RequestSpec(
                request_id=str(values.get("request_id", "")),
                arrival_ns=_number(values.get("arrival_ns", 0.0), "request arrival_ns"),
                prompt_tokens=_integer(values.get("prompt_tokens", 0), "request prompt_tokens"),
                output_tokens=_integer(values.get("output_tokens", 0), "request output_tokens"),
                priority=_integer(values.get("priority", 0), "request priority"),
                deadline_ns=(
                    None
                    if values.get("deadline_ns") is None
                    else _number(values.get("deadline_ns"), "request deadline_ns")
                ),
                metadata=_mapping(values.get("metadata", {}), "request metadata"),
                schema_version=_schema_version(
                    values.get("schema_version", schema_version),
                    "request schema_version",
                ),
            )
        )

    scheduler_raw = data.get("scheduler")
    if scheduler_raw is None:
        raise ValueError("V4 workload.scheduler is required")
    scheduler = scheduler_from_dict(_mapping(scheduler_raw, "scheduler"))

    mtp_raw = data.get("mtp")
    mtp: Optional[MTPPolicy] = None
    if mtp_raw is not None:
        mtp = mtp_from_dict(_mapping(mtp_raw, "mtp"))

    return WorkloadSpec(
        name=str(data.get("name", "")),
        requests=tuple(requests),
        request_count=_integer(data.get("request_count", 1), "request_count"),
        prompt_tokens=_integer(data.get("prompt_tokens", 0), "prompt_tokens"),
        output_tokens=_integer(data.get("output_tokens", 0), "output_tokens"),
        arrival_rate_rps=_number(data.get("arrival_rate_rps", 0.0), "arrival_rate_rps"),
        random_seed=_integer(data.get("random_seed", 0), "random_seed"),
        scheduler=scheduler,
        mtp=mtp,
        metadata=_mapping(data.get("metadata", {}), "workload metadata"),
        schema_version=schema_version,
    )


def _required_profile(
    profiles: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    if name not in profiles:
        raise ValueError("V4 profiles.{} is required".format(name))
    return _mapping(profiles[name], "{} profile".format(name))


def _scenario_with_hardware_input(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Make the explicit hardware input the authority for hardware sections."""
    hardware_input = data.get("hardware_input")
    if hardware_input is None:
        return data
    raw = _mapping(hardware_input, "hardware_input")
    allowed = ("schema_version", "kind", "contract_version", "hardware", "metadata")
    _reject_unknown_fields(raw, "hardware_input", allowed)
    kind = str(raw.get("kind", "hardware_input"))
    if kind != "hardware_input":
        raise ValueError("hardware_input.kind must be hardware_input")
    version = str(raw.get("schema_version", SCHEMA_VERSION))
    if version not in {SCHEMA_VERSION, "4.0"}:
        raise ValueError("hardware_input schema_version must be 4.0 or 4.0.0")
    hardware = dict(_mapping(raw.get("hardware"), "hardware_input.hardware"))
    contract_version = str(raw.get("contract_version", ""))
    if contract_version != "2":
        raise ValueError("hardware_input.contract_version must be 2")
    parameters = _mapping(
        hardware.pop("parameters", None), "hardware_input.hardware.parameters"
    )
    input_profiles = dict(parameters)
    for required in ("host_orchestration", "fusion", "runtime"):
        if required not in input_profiles or input_profiles[required] is None:
            raise ValueError(
                "hardware_input.hardware.parameters.{} is required; refusing to regenerate hardware defaults".format(required)
            )
    gpu_ids = {
        str(item.get("component_id"))
        for item in _array(hardware.get("components", []), "hardware_input.hardware.components")
        if isinstance(item, Mapping)
        and normalize_cost_profile_kind(str(item.get("kind", ""))) == "gpu"
    }
    runtime_raw = _mapping(input_profiles["runtime"], "hardware_input.hardware.parameters.runtime")
    controller_ids = set(_mapping(runtime_raw.get("gpu_controllers"), "runtime.gpu_controllers"))
    if controller_ids != gpu_ids:
        raise ValueError(
            "hardware_input runtime.gpu_controllers must exactly match GPU component IDs"
        )
    inline_registries: Dict[str, Dict[str, Any]] = {
        kind: {} for kind in _COMPONENT_PROFILE_TYPES
    }
    raw_components = _array(hardware.get("components", []), "hardware_input.hardware.components")
    normalized_components = []
    for index, raw_component in enumerate(raw_components):
        component = dict(_mapping(raw_component, "hardware input component"))
        inline = component.pop("execution_profile", None)
        if inline is None:
            component_kind = str(component.get("kind", "")).lower()
            component_metadata = component.get("metadata", {})
            hbf_is_active_memory = (
                component_kind == "hbf"
                and isinstance(component_metadata, Mapping)
                and component_metadata.get("access_mode") == "memory"
            )
            if normalize_cost_profile_kind(component_kind) is None and not hbf_is_active_memory:
                normalized_components.append(component)
                continue
            raise ValueError(
                "hardware_input.hardware.components[{}].execution_profile is required".format(index)
            )
        inline_object = _mapping(inline, "execution_profile")
        profile_id = str(inline_object.get("profile_id", "")).strip()
        profile_kind = str(inline_object.get("profile_kind", "")).strip()
        profile_data = inline_object.get("parameters")
        if not profile_id or not profile_kind or not isinstance(profile_data, Mapping):
            raise ValueError(
                "hardware_input execution_profile requires profile_id, profile_kind, and parameters"
            )
        normalized_kind = normalize_cost_profile_kind(str(component.get("kind", "")))
        if not profile_kind:
            profile_kind = normalized_kind or profile_kind
        if normalized_kind is not None and normalized_kind != profile_kind:
            raise ValueError(
                "hardware_input execution_profile kind does not match component {}".format(
                    component.get("component_id", index)
                )
            )
        if profile_kind is None and str(component.get("kind", "")).lower() == "hbf":
            profile_kind = "host_memory"
        if profile_kind not in _COMPONENT_PROFILE_TYPES:
            raise ValueError(
                "hardware input component {} has no typed execution profile kind".format(
                    component.get("component_id", index)
                )
            )
        component["cost_profile_id"] = profile_id
        profile_data = dict(profile_data)
        existing = inline_registries[profile_kind].get(profile_id)
        if existing is not None and existing != profile_data:
            raise ValueError(
                "hardware_input execution_profile {}.{} is inconsistent across components".format(
                    profile_kind, profile_id
                )
            )
        inline_registries[profile_kind][profile_id] = profile_data
        normalized_components.append(component)
    hardware["components"] = normalized_components
    input_profiles["components"] = inline_registries
    scenario = dict(data)
    scenario.pop("hardware_input", None)
    scenario["hardware"] = dict(hardware)
    raw_scenario_profiles = scenario.get("profiles")
    profiles = (
        dict(_mapping(raw_scenario_profiles, "profiles"))
        if raw_scenario_profiles is not None
        else {}
    )
    for field_name in _HARDWARE_INPUT_PROFILE_FIELDS:
        if field_name in input_profiles:
            profiles[field_name] = input_profiles[field_name]
    scenario["profiles"] = profiles
    return scenario


def _cache_level_profile_from_dict(
    data: Mapping[str, Any], field_name: str
) -> CacheLevelProfile:
    _reject_dataclass_unknown_fields(data, field_name, CacheLevelProfile)
    values = dict(data)
    for name in (
        "capacity_bytes",
        "line_bytes",
        "associativity",
        "banks",
        "read_ports",
        "write_ports",
        "max_outstanding",
    ):
        if name in values:
            values[name] = _integer(
                values[name], "{} {}".format(field_name, name)
            )
    for name in (
        "hit_latency_ns",
        "bandwidth_gb_s",
        "energy_pj_per_byte",
    ):
        if name in values:
            values[name] = _number(
                values[name], "{} {}".format(field_name, name)
            )
    return CacheLevelProfile(**values)


def _cache_hierarchy_profile_from_dict(
    data: Mapping[str, Any], field_name: str
) -> CacheHierarchyProfile:
    _reject_dataclass_unknown_fields(data, field_name, CacheHierarchyProfile)
    raw_levels = _array(data.get("levels"), "{} levels".format(field_name))
    levels = tuple(
        _cache_level_profile_from_dict(
            _mapping(raw, "{} level".format(field_name)),
            "{} level {}".format(field_name, index),
        )
        for index, raw in enumerate(raw_levels)
    )
    return CacheHierarchyProfile(
        levels=levels,
        write_back=_boolean(
            data.get("write_back", True), "{} write_back".format(field_name)
        ),
        write_allocate=_boolean(
            data.get("write_allocate", True),
            "{} write_allocate".format(field_name),
        ),
    )


def _tensor_core_profile_from_dict(
    data: Mapping[str, Any]
) -> TensorCoreProfile:
    _reject_dataclass_unknown_fields(
        data,
        "profiles.components.gpu.*.tensor_core",
        TensorCoreProfile,
    )
    values = dict(data)
    for name in (
        "sm_count",
        "tensor_cores_per_sm",
        "mma_m",
        "mma_n",
        "mma_k",
    ):
        if name in values:
            values[name] = _integer(values[name], "tensor_core {}".format(name))
    for name in ("frequency_ghz", "cycles_per_mma"):
        if name in values:
            values[name] = _number(values[name], "tensor_core {}".format(name))
    if "supported_dtypes" in values:
        values["supported_dtypes"] = tuple(
            str(item)
            for item in _array(
                values["supported_dtypes"], "tensor_core supported_dtypes"
            )
        )
    if "dtype_throughput_scale" in values:
        raw_scales = _mapping(
            values["dtype_throughput_scale"],
            "tensor_core dtype_throughput_scale",
        )
        values["dtype_throughput_scale"] = {
            str(name): _number(
                scale,
                "tensor_core dtype_throughput_scale {}".format(name),
            )
            for name, scale in raw_scales.items()
        }
    return TensorCoreProfile(**values)


def _host_gemm_offload_capability_from_dict(
    data: Mapping[str, Any]
) -> HostGemmOffloadCapability:
    _reject_dataclass_unknown_fields(
        data,
        "profiles.components.gpu.*.host_gemm_offload",
        HostGemmOffloadCapability,
    )
    values = dict(data)
    if "minimum_m" in values:
        values["minimum_m"] = _integer(
            values["minimum_m"],
            "GPU host_gemm_offload minimum_m",
        )
    return HostGemmOffloadCapability(**values)


def _host_recurrent_offload_capability_from_dict(
    data: Mapping[str, Any]
) -> HostRecurrentOffloadCapability:
    field_name = "profiles.components.gpu.*.host_recurrent_offload"
    _reject_dataclass_unknown_fields(
        data,
        field_name,
        HostRecurrentOffloadCapability,
    )
    values = dict(data)
    for name in (
        "minimum_m",
        "query_width",
        "key_width",
        "value_width",
        "conv_kernel_size",
    ):
        if name in values:
            values[name] = _integer(
                values[name], "GPU host_recurrent_offload {}".format(name)
            )
    if "supported_ops" in values:
        raw_ops = _array(
            values["supported_ops"],
            "GPU host_recurrent_offload supported_ops",
        )
        if not all(isinstance(value, str) for value in raw_ops):
            raise ValueError(
                "GPU host_recurrent_offload supported_ops must contain text"
            )
        values["supported_ops"] = tuple(raw_ops)
    return HostRecurrentOffloadCapability(**values)


def _gpu_quantized_matmul_capability_from_dict(
    data: Mapping[str, Any]
) -> GPUQuantizedMatmulCapability:
    field_name = "profiles.components.gpu.*.quantized_matmul_capabilities[]"
    _reject_dataclass_unknown_fields(
        data,
        field_name,
        GPUQuantizedMatmulCapability,
    )
    values = dict(data)
    if "supported_weight_formats" in values:
        raw_formats = _array(
            values["supported_weight_formats"],
            "{} supported_weight_formats".format(field_name),
        )
        if not all(isinstance(value, str) for value in raw_formats):
            raise ValueError(
                "{} supported_weight_formats must contain text".format(
                    field_name
                )
            )
        values["supported_weight_formats"] = tuple(raw_formats)
    if "source_activation_bits" in values:
        values["source_activation_bits"] = tuple(
            _integer(value, "{} source_activation_bits value".format(field_name))
            for value in _array(
                values["source_activation_bits"],
                "{} source_activation_bits".format(field_name),
            )
        )
    for name in ("internal_activation_bits", "accumulator_bits", "min_m"):
        if name in values:
            values[name] = _integer(
                values[name], "{} {}".format(field_name, name)
            )
    return GPUQuantizedMatmulCapability(**values)


def _gpu_profile_from_dict(data: Mapping[str, Any]) -> GPUProfile:
    _reject_dataclass_unknown_fields(
        data,
        "profiles.components.gpu.*",
        GPUProfile,
    )
    values = dict(data)
    values["tensor_core"] = _tensor_core_profile_from_dict(
        _mapping(values.get("tensor_core"), "GPU tensor_core")
    )
    values["cache_hierarchy"] = _cache_hierarchy_profile_from_dict(
        _mapping(values.get("cache_hierarchy"), "GPU cache_hierarchy"),
        "GPU cache_hierarchy",
    )
    if values.get("host_gemm_offload") is not None:
        values["host_gemm_offload"] = (
            _host_gemm_offload_capability_from_dict(
                _mapping(
                    values["host_gemm_offload"],
                    "GPU host_gemm_offload",
                )
            )
        )
    if values.get("host_recurrent_offload") is not None:
        values["host_recurrent_offload"] = (
            _host_recurrent_offload_capability_from_dict(
                _mapping(
                    values["host_recurrent_offload"],
                    "GPU host_recurrent_offload",
                )
            )
        )
    if "quantized_matmul_capabilities" in values:
        values["quantized_matmul_capabilities"] = tuple(
            _gpu_quantized_matmul_capability_from_dict(
                _mapping(raw, "GPU quantized_matmul_capabilities value")
            )
            for raw in _array(
                values["quantized_matmul_capabilities"],
                "GPU quantized_matmul_capabilities",
            )
        )
    for name in (
        "scalar_lanes_per_sm",
        "special_function_units_per_sm",
    ):
        if name in values:
            values[name] = _integer(values[name], "GPU profile {}".format(name))
    for name in (
        "scalar_ops_per_cycle",
        "reduction_ops_per_cycle_per_sm",
        "special_function_ops_per_cycle",
        "occupancy",
        "attainable_efficiency",
        "kernel_launch_ns",
        "tensor_energy_pj_per_op",
        "scalar_energy_pj_per_op",
        "special_function_energy_pj_per_op",
        "launch_energy_pj",
    ):
        if name in values:
            values[name] = _number(values[name], "GPU profile {}".format(name))
    return GPUProfile(**values)


def _cpu_pipeline_profile_from_dict(
    data: Mapping[str, Any]
) -> CPUPipelineProfile:
    _reject_dataclass_unknown_fields(
        data,
        "profiles.components.cpu.*.pipeline",
        CPUPipelineProfile,
    )
    values = dict(data)
    for name in (
        "core_count",
        "simd_width_bits",
        "decode_width",
        "issue_width",
        "retire_width",
        "vector_fma_units_per_core",
        "vector_alu_units_per_core",
        "load_units_per_core",
        "store_units_per_core",
        "branch_units_per_core",
        "special_function_units_per_core",
        "reorder_buffer_entries",
        "load_store_queue_entries",
        "memory_level_parallelism",
    ):
        if name in values:
            values[name] = _integer(
                values[name], "CPU pipeline {}".format(name)
            )
    for name in (
        "frequency_ghz",
        "special_function_cycles_per_vector",
        "branch_mispredict_ns",
    ):
        if name in values:
            values[name] = _number(
                values[name], "CPU pipeline {}".format(name)
            )
    return CPUPipelineProfile(**values)


def _cpu_quantized_dot_capability_from_dict(
    data: Mapping[str, Any]
) -> CPUQuantizedDotCapability:
    field_name = "profiles.components.cpu.*.quantized_dot_capabilities[]"
    _reject_dataclass_unknown_fields(
        data,
        field_name,
        CPUQuantizedDotCapability,
    )
    values = dict(data)
    if "supported_weight_formats" in values:
        raw_formats = _array(
            values["supported_weight_formats"],
            "{} supported_weight_formats".format(field_name),
        )
        if not all(isinstance(value, str) for value in raw_formats):
            raise ValueError(
                "{} supported_weight_formats must contain text".format(
                    field_name
                )
            )
        values["supported_weight_formats"] = tuple(raw_formats)
    if "source_activation_bits" in values:
        values["source_activation_bits"] = tuple(
            _integer(value, "{} source_activation_bits value".format(field_name))
            for value in _array(
                values["source_activation_bits"],
                "{} source_activation_bits".format(field_name),
            )
        )
    for name in (
        "dot_activation_bits",
        "dot_weight_bits",
        "accumulator_bits",
        "activation_quantization_block_elements",
        "maximum_m",
    ):
        if name in values:
            values[name] = _integer(
                values[name], "{} {}".format(field_name, name)
            )
    for name in (
        "effective_ops_per_instruction",
        "dot_issue_instructions_per_cycle_per_core",
        "auxiliary_ops_per_instruction",
    ):
        if name in values:
            values[name] = _number(
                values[name], "{} {}".format(field_name, name)
            )
    if (
        "activation_quantization_instructions_per_block" in values
        and values["activation_quantization_instructions_per_block"] is not None
    ):
        values["activation_quantization_instructions_per_block"] = _number(
            values["activation_quantization_instructions_per_block"],
            "{} activation_quantization_instructions_per_block".format(
                field_name
            ),
        )
    return CPUQuantizedDotCapability(**values)


def _cpu_profile_from_dict(data: Mapping[str, Any]) -> CPUProfile:
    _reject_dataclass_unknown_fields(
        data,
        "profiles.components.cpu.*",
        CPUProfile,
    )
    values = dict(data)
    values["pipeline"] = _cpu_pipeline_profile_from_dict(
        _mapping(values.get("pipeline"), "CPU pipeline")
    )
    values["cache_hierarchy"] = _cache_hierarchy_profile_from_dict(
        _mapping(values.get("cache_hierarchy"), "CPU cache_hierarchy"),
        "CPU cache_hierarchy",
    )
    if "quantized_dot_capabilities" in values:
        values["quantized_dot_capabilities"] = tuple(
            _cpu_quantized_dot_capability_from_dict(
                _mapping(raw, "CPU quantized_dot_capabilities value")
            )
            for raw in _array(
                values["quantized_dot_capabilities"],
                "CPU quantized_dot_capabilities",
            )
        )
    for name in (
        "attainable_efficiency",
        "dispatch_ns",
        "gemm_energy_pj_per_op",
        "elementwise_energy_pj_per_op",
        "reduction_energy_pj_per_op",
        "special_function_energy_pj_per_op",
        "dispatch_energy_pj",
    ):
        if name in values:
            values[name] = _number(values[name], "CPU profile {}".format(name))
    return CPUProfile(**values)


def _host_orchestration_profile_from_dict(
    data: Mapping[str, Any]
) -> HostOrchestrationProfile:
    _reject_dataclass_unknown_fields(
        data, "profiles.host_orchestration", HostOrchestrationProfile
    )
    values = dict(data)
    for name in (
        "capacity_fixed_instructions",
        "capacity_instructions_per_request",
        "schedule_fixed_instructions",
        "schedule_instructions_per_request",
        "schedule_instructions_per_token",
        "command_build_fixed_instructions",
        "command_build_instructions_per_invocation",
        "descriptor_bytes_per_request",
        "token_bytes",
        "max_inflight_batches",
        "kv_descriptor_bytes",
    ):
        if name in values:
            values[name] = _integer(
                values[name], "host orchestration {}".format(name)
            )
    for name in (
        "request_parse_ns",
        "batch_fixed_ns",
        "token_pack_ns",
        "submission_ns",
        "dma_queue_submission_ns",
        "dma_bandwidth_gb_s",
        "dma_latency_ns",
        "kv_page_lookup_ns",
        "kv_descriptor_ns",
        "admission_ns",
        "input_decode_ns_per_token",
        "output_encode_ns_per_token",
    ):
        if name in values:
            values[name] = _number(
                values[name], "host orchestration {}".format(name)
            )
    if "pinned_memory" in values:
        values["pinned_memory"] = _boolean(
            values["pinned_memory"], "host orchestration pinned_memory"
        )
    return HostOrchestrationProfile(**values)


def _fusion_policy_from_dict(data: Mapping[str, Any]) -> FusionPolicy:
    _reject_dataclass_unknown_fields(data, "profiles.fusion", FusionPolicy)
    required_flags = (
        "qkv_rope",
        "flash_attention",
        "gemm_epilogue_activation",
        "residual_norm",
    )
    missing = tuple(name for name in required_flags if name not in data)
    if missing:
        raise ValueError(
            "V4 fusion profile is missing flags: {}".format(
                ", ".join(missing)
            )
        )
    return FusionPolicy(
        qkv_rope=_boolean(data["qkv_rope"], "fusion qkv_rope"),
        flash_attention=_boolean(
            data["flash_attention"], "fusion flash_attention"
        ),
        gemm_epilogue_activation=_boolean(
            data["gemm_epilogue_activation"],
            "fusion gemm_epilogue_activation",
        ),
        residual_norm=_boolean(
            data["residual_norm"], "fusion residual_norm"
        ),
        max_fused_working_set_bytes=_integer(
            data.get("max_fused_working_set_bytes", 0),
            "fusion max_fused_working_set_bytes",
        ),
    )


def _hbm_profile_from_dict(data: Mapping[str, Any]) -> HBMProfile:
    values = dict(data)
    _reject_dataclass_unknown_fields(values, "component HBM profile", HBMProfile)
    for field_name in ("bandwidth_gb_s", "efficiency", "measured_effective_bandwidth_gb_s", "energy_pj_per_byte",
                       "read_latency_ns", "write_latency_ns"):
        if field_name in values:
            values[field_name] = _number(
                values[field_name], "HBM profile {}".format(field_name)
            )
    for field_name in ("transaction_bytes", "max_outstanding_requests"):
        if field_name in values:
            values[field_name] = _integer(values[field_name], "HBM profile {}".format(field_name))
    for field_name in ("read_bandwidth_gb_s", "write_bandwidth_gb_s"):
        if field_name in values and values[field_name] is not None:
            values[field_name] = _number(values[field_name], "HBM profile {}".format(field_name))
    return HBMProfile(**values)


def _host_memory_profile_from_dict(
    data: Mapping[str, Any],
) -> HostMemoryProfile:
    values = dict(data)
    _reject_dataclass_unknown_fields(
        values, "component host-memory profile", HostMemoryProfile
    )
    for field_name in ("bandwidth_gb_s", "efficiency", "measured_effective_bandwidth_gb_s", "energy_pj_per_byte",
                       "read_latency_ns", "write_latency_ns"):
        if field_name in values:
            values[field_name] = _number(
                values[field_name],
                "host-memory profile {}".format(field_name),
            )
    for field_name in ("transaction_bytes", "max_outstanding_requests"):
        if field_name in values:
            values[field_name] = _integer(values[field_name], "host-memory profile {}".format(field_name))
    for field_name in ("read_bandwidth_gb_s", "write_bandwidth_gb_s"):
        if field_name in values and values[field_name] is not None:
            values[field_name] = _number(values[field_name], "host-memory profile {}".format(field_name))
    return HostMemoryProfile(**values)


def _cim_profile_from_dict(
    data: Mapping[str, Any],
) -> DigitalSramCimProfile:
    values = dict(data)
    _reject_dataclass_unknown_fields(
        values, "component CIM profile", DigitalSramCimProfile
    )
    integer_fields = {
        "conversion_scratch_capacity_bytes",
        "tile_m",
        "tile_k",
        "tile_n",
        "array_count",
        "p_m",
        "p_k",
        "p_n",
        "input_parallel_bits",
        "weight_parallel_bits",
        "cycles_per_eval",
        "weight_capacity_bytes",
        "max_m_replication",
        "noc_reduce_fan_in",
        "accumulator_bits",
        "accumulator_guard_bits",
    }
    numeric_fields = {
        "weight_decode_elements_per_ns",
        "activation_fp32_to_fp16_elements_per_ns",
        "conversion_read_energy_pj_per_byte",
        "conversion_write_energy_pj_per_byte",
        "frequency_ghz",
        "load_bandwidth_gb_s",
        "activation_bandwidth_gb_s",
        "output_bandwidth_gb_s",
        "noc_bandwidth_gb_s",
        "accumulator_outputs_per_cycle",
        "float_cycles_per_eval",
        "float_accumulator_outputs_per_cycle",
        "peripheral_elements_per_cycle",
        "load_latency_ns",
        "noc_hop_latency_ns",
        "peripheral_latency_ns",
        "eval_energy_pj",
        "load_energy_pj_per_byte",
        "activation_energy_pj_per_byte",
        "output_energy_pj_per_byte",
        "noc_energy_pj_per_byte",
        "accumulator_energy_pj_per_op",
        "peripheral_energy_pj_per_element",
    }
    for field_name in integer_fields.intersection(values):
        values[field_name] = _integer(
            values[field_name], "CIM profile {}".format(field_name)
        )
    for field_name in numeric_fields.intersection(values):
        if field_name in ("float_cycles_per_eval", "float_accumulator_outputs_per_cycle") and values[field_name] is None:
            continue
        values[field_name] = _number(
            values[field_name], "CIM profile {}".format(field_name)
        )
    for field_name in ("supported_activation_bits", "supported_weight_bits"):
        if field_name in values:
            values[field_name] = tuple(
                _integer(value, "CIM profile {} value".format(field_name))
                for value in _array(values[field_name], field_name)
            )
    return DigitalSramCimProfile(**values)


def _interconnect_profile_from_dict(
    data: Mapping[str, Any],
) -> InterconnectProfile:
    values = dict(data)
    _reject_dataclass_unknown_fields(
        values, "profiles.cim_interconnect", InterconnectProfile
    )
    for field_name in ("bandwidth_gb_s", "latency_ns", "energy_pj_per_byte"):
        if field_name in values:
            values[field_name] = _number(
                values[field_name],
                "CIM interconnect {}".format(field_name),
            )
    return InterconnectProfile(**values)


def _component_profile_from_dict(
    kind: str, data: Mapping[str, Any]
) -> ComponentCostProfile:
    if kind == "gpu":
        return _gpu_profile_from_dict(data)
    if kind == "hbm":
        return _hbm_profile_from_dict(data)
    if kind == "cpu":
        return _cpu_profile_from_dict(data)
    if kind == "host_memory":
        return _host_memory_profile_from_dict(data)
    if kind == "cim":
        return _cim_profile_from_dict(data)
    raise ValueError("unknown component profile kind: {}".format(kind))


def _component_profile_registries_from_dict(
    data: Mapping[str, Any],
) -> Mapping[str, Mapping[str, ComponentCostProfile]]:
    _reject_unknown_fields(
        data,
        "profiles.components",
        tuple(_COMPONENT_PROFILE_TYPES),
    )
    result: Dict[str, Mapping[str, ComponentCostProfile]] = {}
    for kind, raw_registry in data.items():
        registry = _mapping(
            raw_registry,
            "profiles.components.{}".format(kind),
        )
        parsed: Dict[str, ComponentCostProfile] = {}
        for raw_profile_id, raw_profile in registry.items():
            profile_id = str(raw_profile_id)
            if not profile_id.strip():
                raise ValueError("component profile id must not be empty")
            parsed[profile_id] = _component_profile_from_dict(
                kind,
                _mapping(
                    raw_profile,
                    "profiles.components.{}.{}".format(kind, profile_id),
                ),
            )
        result[kind] = parsed
    return result


def _seed_omitted_memory_profile_bandwidths(
    data: Mapping[str, Any], hardware: HardwareSpec
) -> Mapping[str, Any]:
    """Fill only omitted memory profile rates from authored hardware caps.

    An empty ``profiles.components.*`` entry is common in hand-authored JSON.
    The dataclass defaults (1 GB/s) must not turn that omission into a real
    one-GB/s calibration.  Explicit profile rates, including directional and
    measured values, are left untouched.
    """

    result = {kind: dict(registry) for kind, registry in data.items()}
    components = tuple(hardware.components)
    for profile_kind in ("hbm", "host_memory"):
        registry = result.get(profile_kind)
        if not isinstance(registry, dict):
            continue
        for profile_id, raw_profile in tuple(registry.items()):
            if not isinstance(raw_profile, Mapping):
                continue
            matched = tuple(
                item for item in components
                if item.cost_profile_id == str(profile_id)
                and normalize_cost_profile_kind(item.normalized_kind) == profile_kind
            )
            if not matched:
                continue
            payload = dict(raw_profile)
            if "bandwidth_gb_s" not in payload and "measured_effective_bandwidth_gb_s" not in payload:
                aggregate = any(
                    str(item.metadata.get("memory_bandwidth_scope", "")).strip().lower() == "aggregate"
                    for item in matched
                )
                if aggregate:
                    owners = {
                        str(item.metadata.get("memory_aggregate_owner", "")).strip()
                        for item in matched
                        if str(item.metadata.get("memory_aggregate_owner", "")).strip()
                    }
                    source = tuple(
                        item for item in components
                        if item.is_active_memory
                        and str(item.metadata.get("memory_bandwidth_scope", "")).strip().lower() == "aggregate"
                        and str(item.metadata.get("memory_aggregate_owner", "")).strip() in owners
                    )
                    source = source or matched
                    bandwidth_gbps = sum(item.shared_bandwidth_gbps for item in source)
                else:
                    bandwidth_gbps = max(item.shared_bandwidth_gbps for item in matched)
                if bandwidth_gbps > 0:
                    payload["bandwidth_gb_s"] = bandwidth_gbps / 8.0
                directional = any(
                    str(item.metadata.get("bandwidth_mode", "")).strip().lower()
                    == "directional"
                    for item in matched
                )
                if directional:
                    for direction in ("read", "write"):
                        values = [
                            item.directional_bandwidth_gbps(direction)
                            for item in matched
                            if item.directional_bandwidth_gbps(direction) > 0
                        ]
                        if values:
                            value = sum(values) if aggregate else max(values)
                            payload[direction + "_bandwidth_gb_s"] = value / 8.0
            if "resource_id" not in payload:
                owners = {
                    str(item.metadata.get("memory_service_owner", "")).strip()
                    for item in matched
                    if str(item.metadata.get("memory_service_owner", "")).strip()
                }
                if len(owners) == 1:
                    payload["resource_id"] = next(iter(owners))
            registry[profile_id] = payload
    return result


def _host_output_contract_from_dict(
    data: Mapping[str, Any],
) -> HostOutputContract:
    _reject_dataclass_unknown_fields(
        data, "profiles.host_output", HostOutputContract
    )
    return HostOutputContract(
        target_component_id=str(data.get("target_component_id", "")),
        vocabulary_size=_integer(
            data.get("vocabulary_size", 0),
            "profiles.host_output vocabulary_size",
        ),
        logits_dtype=str(data.get("logits_dtype", "")),
        logits_element_bytes=_integer(
            data.get("logits_element_bytes", 0),
            "profiles.host_output logits_element_bytes",
        ),
        allocation_semantics=str(
            data.get("allocation_semantics", "unspecified")
        ),
    )


def _sampling_policy_from_dict(data: Mapping[str, Any]) -> SamplingPolicy:
    _reject_dataclass_unknown_fields(data, "profiles.sampling", SamplingPolicy)
    return SamplingPolicy(
        mode=str(data.get("mode", "")),
        temperature=(
            None
            if data.get("temperature") is None
            else _number(
                data.get("temperature"), "profiles.sampling temperature"
            )
        ),
        implementation=(
            None
            if data.get("implementation") is None
            else str(data.get("implementation"))
        ),
        top_k=(
            None
            if data.get("top_k") is None
            else _integer(data.get("top_k"), "profiles.sampling top_k")
        ),
        top_p=(
            None
            if data.get("top_p") is None
            else _number(data.get("top_p"), "profiles.sampling top_p")
        ),
        min_p=(
            None
            if data.get("min_p") is None
            else _number(data.get("min_p"), "profiles.sampling min_p")
        ),
        min_keep=(
            None
            if data.get("min_keep") is None
            else _integer(
                data.get("min_keep"), "profiles.sampling min_keep"
            )
        ),
    )


def _llama_cpp_config_from_dict(data: Mapping[str, Any]) -> LlamaCppRuntimeConfig:
    """Parse the typed llama.cpp runtime profile from a scenario document."""

    _reject_dataclass_unknown_fields(data, "profiles.llama_cpp", LlamaCppRuntimeConfig)
    return LlamaCppRuntimeConfig(**dict(data))


def _bind_local_rtx5080_hardware_presets(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve the local RTX5080 scene through the curated hardware catalog.

    The native parity scene is a measured host/GPU snapshot, but its historical
    ``hbm0`` node is a GPU-local memory modeling endpoint rather than physical
    HBM.  Keep the measured fields under
    ``scenario_observed_hardware`` and make the selected catalog component the
    authoritative capability source for the simulation.
    """

    hardware = data.get("hardware")
    if not isinstance(hardware, Mapping):
        return data
    # A frontend architecture-preset import is already fully bound to the
    # curated component catalog.  Do not reinterpret it as the legacy native
    # snapshot merely because the scenario name still contains ``rtx5080``.
    architecture_metadata = hardware.get("metadata", {})
    architecture_preset = (
        architecture_metadata.get("architecture_preset")
        if isinstance(architecture_metadata, Mapping)
        else None
    )
    if isinstance(architecture_preset, Mapping) and str(architecture_preset.get("id", "")).strip():
        return data
    # Native parity binding is an explicit import contract.  Names are user
    # labels and must never silently replace authored hardware.
    hardware_metadata = hardware.get("metadata", {})
    marker = (
        hardware_metadata.get("native_hardware_preset_id")
        if isinstance(hardware_metadata, Mapping)
        else None
    )
    if marker != "local-rtx5080-9950x3d":
        return data
    # A native parity builder may already have materialized the curated GPU,
    # CPU, host-memory, and attached-memory bindings.  Rebinding that complete
    # authoring payload would replace its calibrated profile resource IDs and
    # latency fields on every frontend round trip.  Only bind raw snapshots
    # that lack these explicit component-level references.
    raw_components = hardware.get("components", ())
    complete_binding = True
    for raw_component in raw_components if isinstance(raw_components, (list, tuple)) else ():
        if not isinstance(raw_component, Mapping):
            complete_binding = False
            break
        kind = normalize_component_kind(str(raw_component.get("kind", "")))
        metadata = raw_component.get("metadata", {})
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if kind == "hbm":
            bound = metadata.get("attached_memory_preset_id")
        elif kind in {"gpu", "cpu", "host_memory", "dram"}:
            bound = metadata.get("component_preset_id")
        else:
            continue
        if not isinstance(bound, str) or not bound.strip():
            complete_binding = False
            break
    if complete_binding and raw_components:
        return data
    from .component_presets import materialize_component_payload

    preset_by_kind = {
        "cpu": "amd-ryzen-9-9950x3d",
        "gpu": "nvidia-rtx-5080",
        "host_memory": "acer-local-ddr5-128gb-5600-dual-channel",
        "dram": "acer-local-ddr5-128gb-5600-dual-channel",
        "hbm": "gddr7-16gb-30_0-256bit",
    }
    payloads = {
        preset_id: materialize_component_payload(preset_id)
        for preset_id in set(preset_by_kind.values())
    }
    components = []
    preset_by_component: Dict[str, Mapping[str, Any]] = {}
    for raw_component in hardware.get("components", ()):
        component = dict(raw_component)
        kind = normalize_component_kind(str(component.get("kind", "")))
        preset_id = preset_by_kind.get(kind)
        if not preset_id:
            components.append(component)
            continue
        preset = payloads[preset_id]
        observed = dict(component.get("metadata", {}))
        metadata = dict(preset.get("metadata", {}))
        metadata["scenario_observed_hardware"] = observed
        metadata["component_preset_id"] = preset_id
        metadata["component_preset_status"] = "catalog_reference"
        if kind == "hbm":
            # native_llama_compare uses hbm0 as the simulator's local GPU
            # memory service endpoint.  The native hardware is RTX5080 GDDR7,
            # so bind that logical endpoint to the explicit GDDR7 preset and
            # keep it marked as attached memory rather than physical HBM.
            metadata.pop("component_preset_id", None)
            metadata["attached_memory_preset_id"] = preset_id
            metadata["component_preset_status"] = "logical_gpu_attached_memory"
            metadata["physical_hardware_component"] = False
            technology = dict(preset.get("metadata", {}).get("technology", {}))
            metadata["memory_type"] = technology.get("memory_type", kind.upper())
        if kind in {"hbm", "host_memory", "dram"}:
            metadata["memory_service_owner"] = (
                "{}.hbm_fabric".format(component.get("component_id"))
                if kind == "hbm"
                else "{}.memory".format(component.get("component_id"))
            )
            metadata["resident_access_path"] = "topology"
        component["metadata"] = metadata
        component["cost_profile_id"] = component.get("cost_profile_id") or preset.get("cost_profile_id")
        preset_by_component[str(component.get("component_id", ""))] = preset
        if kind in {"hbm", "host_memory", "dram"}:
            for field in ("capacity_bytes", "bandwidth_gbps", "read_bandwidth_gbps", "write_bandwidth_gbps"):
                component[field] = preset[field]
            preset_ports = list(preset.get("ports", ()))
            current_ports = list(component.get("ports", ()))
            ports = []
            for index, current in enumerate(current_ports or preset_ports):
                source = dict(preset_ports[min(index, len(preset_ports) - 1)]) if preset_ports else dict(current)
                source["port_id"] = current.get("port_id", source.get("port_id", "host"))
                source["metadata"] = {**dict(source.get("metadata", {})), **dict(current.get("metadata", {}))}
                ports.append(source)
            component["ports"] = ports
        elif kind in {"gpu", "cpu"}:
            ddr_preset_port = dict(next(iter(payloads["acer-local-ddr5-128gb-5600-dual-channel"].get("ports", ())), {}))
            if kind == "cpu" and ddr_preset_port:
                component["ports"] = [
                    {
                        **dict(port),
                        **({
                            "protocol": ddr_preset_port["protocol"],
                            "version": ddr_preset_port["version"],
                            "lanes": ddr_preset_port["lanes"],
                            "bandwidth_gbps": ddr_preset_port["bandwidth_gbps"],
                            "metadata": {**dict(ddr_preset_port.get("metadata", {})), **dict(port.get("metadata", {}))},
                        } if normalize_component_kind(str(port.get("protocol", ""))) in {"ddr", "ddr3", "ddr4", "ddr5"} else {}),
                    }
                    for port in component.get("ports", ())
                ]
            elif kind == "gpu":
                gpu_preset_ports = list(preset.get("ports", ()))
                ports = []
                memory_port_bound = False
                for port in component.get("ports", ()):
                    is_local_memory_port = normalize_component_kind(str(port.get("protocol", ""))) in {"hbm", "gddr7"}
                    if is_local_memory_port:
                        if memory_port_bound or not gpu_preset_ports:
                            # The native snapshot may retain the illustrative
                            # hbm1..hbm7 controller ports.  RTX 5080 has one
                            # GDDR7 interface in this parity scene.
                            continue
                        source = dict(gpu_preset_ports[0])
                        source["port_id"] = port.get("port_id", source.get("port_id", "gddr7"))
                        source["metadata"] = {**dict(source.get("metadata", {})), **dict(port.get("metadata", {}))}
                        ports.append(source)
                        memory_port_bound = True
                    else:
                        ports.append(dict(port))
                component["ports"] = ports
        components.append(component)
    result = dict(data)
    result["hardware"] = dict(hardware)
    result["hardware"]["components"] = components

    profiles = dict(data.get("profiles", {}))
    registries = {key: dict(value) for key, value in dict(profiles.get("components", {})).items()}
    for component in components:
        preset = preset_by_component.get(str(component.get("component_id", "")))
        kind = normalize_component_kind(str(component.get("kind", "")))
        if not preset or kind not in {"hbm", "host_memory", "dram"}:
            continue
        profile_key = "hbm" if kind == "hbm" else "host_memory"
        profile_id = str(component.get("cost_profile_id", ""))
        profile = dict(dict(registries.get(profile_key, {})).get(profile_id, {}))
        template = dict(dict(preset.get("metadata", {})).get("cost_profile_template", {}))
        for field in ("bandwidth_gb_s", "read_latency_ns", "write_latency_ns", "transaction_bytes", "max_outstanding_requests"):
            if field in template:
                profile[field] = template[field]
        component_id = str(component.get("component_id", "memory"))
        profile["resource_id"] = "{}.hbm_fabric".format(component_id) if profile_key == "hbm" else "{}.memory".format(component_id)
        registries.setdefault(profile_key, {})[profile_id] = profile
    profiles["components"] = registries
    result["profiles"] = profiles

    links = []
    for raw_link in hardware.get("links", ()):
        link = dict(raw_link)
        source_id = str(link.get("source_component", ""))
        target_id = str(link.get("target_component", ""))
        memory_id = source_id if source_id in preset_by_component and normalize_component_kind(str(preset_by_component[source_id].get("kind", ""))) in {"hbm", "host_memory", "dram"} else target_id if target_id in preset_by_component and normalize_component_kind(str(preset_by_component[target_id].get("kind", ""))) in {"hbm", "host_memory", "dram"} else ""
        preset = preset_by_component.get(memory_id)
        preset_port = next(iter(preset.get("ports", ())), {}) if preset else {}
        protocol = normalize_component_kind(str(link.get("protocol", "")))
        if preset and protocol in {"hbm", "gddr7", "ddr", "ddr3", "ddr4", "ddr5", "dram"}:
            link["protocol"] = preset_port.get("protocol", link.get("protocol"))
            link["version"] = preset_port.get("version", link.get("version"))
            link["lanes"] = preset_port.get("lanes", link.get("lanes"))
            link["bandwidth_gbps"] = preset_port.get("bandwidth_gbps", link.get("bandwidth_gbps"))
            metadata = dict(link.get("metadata", {}))
            metadata["bandwidth_source"] = "memory_component"
            metadata["bandwidth_resource_id"] = (
                "{}.hbm_fabric".format(memory_id) if normalize_component_kind(str(preset.get("kind", ""))) == "hbm" else "{}.memory".format(memory_id)
            )
            link["metadata"] = metadata
        links.append(link)
    result["hardware"]["links"] = links
    result["hardware"].setdefault("metadata", {})
    result["hardware"]["metadata"] = {
        **dict(result["hardware"]["metadata"]),
        "component_preset_bindings": {
            component_id: str(dict(component.get("metadata", {})).get("component_preset_id"))
            for component_id, component in ((str(item.get("component_id", "")), item) for item in components)
            if dict(component.get("metadata", {})).get("component_preset_id")
        },
        "attached_memory_preset_bindings": {
            component_id: str(dict(component.get("metadata", {})).get("attached_memory_preset_id"))
            for component_id, component in ((str(item.get("component_id", "")), item) for item in components)
            if dict(component.get("metadata", {})).get("attached_memory_preset_id")
        },
    }
    public_gpu_specs = dict(result["hardware"]["metadata"].get("gpu_public_specs", {}))
    public_gpu_memory = dict(public_gpu_specs.get("memory", {}))
    if public_gpu_memory:
        public_gpu_memory["preset_id"] = "gddr7-16gb-30_0-256bit"
        public_gpu_specs["memory"] = public_gpu_memory
        result["hardware"]["metadata"]["gpu_public_specs"] = public_gpu_specs
    public_cpu_specs = dict(result["hardware"]["metadata"].get("cpu_public_specs", {}))
    public_cpu_memory = dict(public_cpu_specs.get("memory", {}))
    if public_cpu_memory:
        public_cpu_memory["preset_id"] = "acer-local-ddr5-128gb-5600-dual-channel"
        public_cpu_specs["memory"] = public_cpu_memory
        result["hardware"]["metadata"]["cpu_public_specs"] = public_cpu_specs
    return result


def _bind_local_memory_link_sources(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Import legacy profile physical fields into the component service contract.

    A profile bandwidth is calibrated throughput, so it is deliberately not
    compared for equality with the component's physical ceiling.
    """
    hardware = data.get("hardware")
    profiles = data.get("profiles")
    if not isinstance(hardware, Mapping) or not isinstance(profiles, Mapping):
        return data
    registries = profiles.get("components", {})
    components = []
    for raw in hardware.get("components", ()):
        component = dict(raw)
        metadata = dict(component.get("metadata", {}))
        kind = normalize_cost_profile_kind(normalize_component_kind(str(component.get("kind", ""))))
        if kind in {"hbm", "host_memory"}:
            profile_id = component.get("cost_profile_id")
            profile = registries.get(kind, {}).get(profile_id, {})
            for profile_name, metadata_name in (
                ("service_model", "memory_service_model"),
                ("read_latency_ns", "read_latency_ns"),
                ("write_latency_ns", "write_latency_ns"),
                ("transaction_bytes", "transfer_granularity_bytes"),
                ("max_outstanding_requests", "max_outstanding_requests"),
            ):
                if profile_name not in profile:
                    continue
                value = profile[profile_name]
                if metadata_name in metadata and metadata[metadata_name] != value:
                    raise ValueError("hardware.components[{}].metadata.{}={} conflicts with profiles.components.{}.{}.{}={}".format(
                        component.get("component_id"), metadata_name, metadata[metadata_name], kind, profile_id, profile_name, value))
            if metadata.get("latency_scope", "memory_service") != "memory_service":
                raise ValueError("hardware.components[{}].metadata.latency_scope must be memory_service; end-to-end latency cannot be added to independent links".format(component.get("component_id")))
        component["metadata"] = metadata
        components.append(component)
    return {**data, "hardware": {**hardware, "components": components}}


def scenario_from_dict(data: Mapping[str, Any]) -> ScenarioConfig:
    data = _scenario_with_hardware_input(data)
    data = _bind_local_rtx5080_hardware_presets(data)
    data = _bind_local_memory_link_sources(data)
    _reject_training_fields(data, "scenario")
    _reject_unknown_fields(
        data,
        "scenario",
        (
            "schema_version",
            "name",
            "hardware",
            "model",
            "placement",
            "workload",
            "profiles",
            "weights_resident",
            "assumptions",
        ),
    )
    if "schema_version" not in data:
        raise ValueError("V4 scenario.schema_version is required")
    scenario_schema_version = _schema_version(
        data["schema_version"], "scenario schema_version"
    )
    section_data: Dict[str, Mapping[str, Any]] = {}
    for section_name in ("hardware", "model", "placement", "workload"):
        values = dict(_mapping(data.get(section_name), section_name))
        values.setdefault("schema_version", scenario_schema_version)
        section_data[section_name] = values
    profiles = _mapping(data.get("profiles"), "profiles")
    _reject_unknown_fields(
        profiles,
        "profiles",
        (
            "components",
            "host_orchestration",
            "fusion",
            "cim_interconnect",
            "runtime",
            "llama_cpp",
            "host_output",
            "sampling",
        ),
    )
    hardware = hardware_from_dict(section_data["hardware"])
    component_profile_payload = _seed_omitted_memory_profile_bandwidths(
        _required_profile(profiles, "components"), hardware
    )
    component_profiles = _component_profile_registries_from_dict(
        component_profile_payload
    )
    host_orchestration_profile = _host_orchestration_profile_from_dict(
        _required_profile(profiles, "host_orchestration")
    )
    fusion_policy = _fusion_policy_from_dict(
        _required_profile(profiles, "fusion")
    )
    runtime_raw = profiles.get("runtime")
    llama_cpp_raw = profiles.get("llama_cpp")
    runtime_profile = (
        runtime_profile_from_dict(
            _mapping(runtime_raw, "profiles.runtime")
        )
        if runtime_raw is not None
        else ControllerProfile.architecture_default(
            section_data["hardware"]
        )
    )
    link_raw = profiles.get("cim_interconnect")
    host_output_raw = profiles.get("host_output")
    sampling_raw = profiles.get("sampling")
    return ScenarioConfig(
        name=str(data.get("name", "")),
        hardware=hardware,
        model=model_from_dict(section_data["model"]),
        placement=placement_from_dict(section_data["placement"]),
        workload=workload_from_dict(section_data["workload"]),
        component_profiles=component_profiles,
        host_orchestration_profile=host_orchestration_profile,
        fusion_policy=fusion_policy,
        host_output_contract=(
            _host_output_contract_from_dict(
                _mapping(host_output_raw, "profiles.host_output")
            )
            if host_output_raw is not None
            else None
        ),
        sampling_policy=(
            _sampling_policy_from_dict(
                _mapping(sampling_raw, "profiles.sampling")
            )
            if sampling_raw is not None
            else None
        ),
        runtime_profile=runtime_profile,
        llama_cpp_config=(
            _llama_cpp_config_from_dict(
                _mapping(llama_cpp_raw, "profiles.llama_cpp")
            )
            if llama_cpp_raw is not None
            else None
        ),
        cim_interconnect=(
            _interconnect_profile_from_dict(
                _mapping(link_raw, "CIM interconnect")
            )
            if link_raw is not None
            else None
        ),
        weights_resident=_boolean(data.get("weights_resident", True), "weights_resident"),
        schema_version=scenario_schema_version,
        assumptions=tuple(
            str(item) for item in _array(data.get("assumptions", []), "assumptions")
        ),
    )


def load_scenario(path: Path) -> ScenarioConfig:
    return scenario_from_dict(read_json(path))


__all__ = [
    "ComponentCostProfile",
    "FusionPolicy",
    "HostOutputContract",
    "InterconnectProfile",
    "SamplingPolicy",
    "ControllerProfile",
    "LlamaCppRuntimeConfig",
    "ScenarioConfig",
    "hardware_from_dict",
    "load_scenario",
    "model_from_dict",
    "mtp_from_dict",
    "parallel_from_dict",
    "placement_from_dict",
    "kv_policy_from_dict",
    "rank_mapping_from_dict",
    "scenario_from_dict",
    "normalize_cost_profile_kind",
    "scheduler_from_dict",
    "workload_from_dict",
]
