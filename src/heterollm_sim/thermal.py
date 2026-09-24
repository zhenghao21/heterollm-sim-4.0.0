"""Opt-in static thermal operating-point derating, never a temperature solver.

Factors are supplied by the caller (measurement or labeled sensitivity input).
No power threshold, temperature rise, cooldown, or calibration is inferred.
Always apply to a baseline scenario: capacities, placement and ownership stay
fixed while cloned profiles and link/endpoint rates change together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from numbers import Real

from .config import ScenarioConfig
from .cost_models import (
    CPUProfile, DigitalSramCimProfile, GPUProfile, HBMProfile, HostMemoryProfile,
)


@dataclass(frozen=True)
class ThermalOperatingPoint:
    domain_id: str
    enabled: bool = False
    frequency_scale: float = 1.0
    memory_bandwidth_scale: float = 1.0
    link_bandwidth_scale: float = 1.0
    latency_scale: float = 1.0
    evidence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.domain_id, str) or not self.domain_id.strip():
            raise ValueError("thermal domain_id must be non-empty text")
        if not isinstance(self.enabled, bool):
            raise ValueError("thermal enabled must be boolean")
        for name in ("frequency_scale", "memory_bandwidth_scale", "link_bandwidth_scale", "latency_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
            if (name == "latency_scale" and value < 1) or (name != "latency_scale" and value > 1):
                raise ValueError(name + " must derate, not accelerate, the baseline")
        if not isinstance(self.evidence, str) or (self.enabled and not self.evidence.strip()):
            raise ValueError("enabled thermal point requires explicit evidence or a labeled analytical assumption")


def _derate_profile(profile, point: ThermalOperatingPoint):
    """Clone known typed profiles, retaining capacities, precision and resource IDs."""

    def cache(hierarchy):
        return replace(hierarchy, levels=tuple(
            replace(level, bandwidth_gb_s=level.bandwidth_gb_s * point.memory_bandwidth_scale,
                    hit_latency_ns=level.hit_latency_ns * point.latency_scale)
            for level in hierarchy.levels
        ))

    if isinstance(profile, GPUProfile):
        return replace(profile,
                       tensor_core=replace(profile.tensor_core, frequency_ghz=profile.tensor_core.frequency_ghz * point.frequency_scale),
                       cache_hierarchy=cache(profile.cache_hierarchy))
    if isinstance(profile, CPUProfile):
        return replace(profile,
                       pipeline=replace(profile.pipeline, frequency_ghz=profile.pipeline.frequency_ghz * point.frequency_scale,
                                        branch_mispredict_ns=profile.pipeline.branch_mispredict_ns * point.latency_scale),
                       cache_hierarchy=cache(profile.cache_hierarchy))
    if isinstance(profile, (HBMProfile, HostMemoryProfile)):
        changes = {"bandwidth_gb_s": profile.bandwidth_gb_s * point.memory_bandwidth_scale}
        # Reference registries may predate the additive read/write timing fields.
        for field in fields(profile):
            if field.name in {"read_latency_ns", "write_latency_ns"}:
                changes[field.name] = getattr(profile, field.name) * point.latency_scale
            elif field.name in {"read_bandwidth_gb_s", "write_bandwidth_gb_s"}:
                value = getattr(profile, field.name)
                if value is not None:
                    changes[field.name] = value * point.memory_bandwidth_scale
        return replace(profile, **changes)
    if isinstance(profile, DigitalSramCimProfile):
        return replace(profile,
                       frequency_ghz=profile.frequency_ghz * point.frequency_scale,
                       load_bandwidth_gb_s=profile.load_bandwidth_gb_s * point.memory_bandwidth_scale,
                       activation_bandwidth_gb_s=profile.activation_bandwidth_gb_s * point.memory_bandwidth_scale,
                       output_bandwidth_gb_s=profile.output_bandwidth_gb_s * point.memory_bandwidth_scale,
                       noc_bandwidth_gb_s=profile.noc_bandwidth_gb_s * point.link_bandwidth_scale,
                       load_latency_ns=profile.load_latency_ns * point.latency_scale,
                       noc_hop_latency_ns=profile.noc_hop_latency_ns * point.latency_scale,
                       peripheral_latency_ns=profile.peripheral_latency_ns * point.latency_scale)
    raise TypeError("unsupported thermal cost profile: " + type(profile).__name__)


def apply_thermal_operating_point(
    scenario: ScenarioConfig, operating_point: ThermalOperatingPoint,
) -> ScenarioConfig:
    """Return a derated copy; disabled is an identity/no-op.

    Domain membership is explicit on components and links via
    ``metadata['thermal_domain_id']``. Profiles shared with another domain are
    cloned per component, never edited in place. Route caps on the associated
    ports change along with links; off-domain host CPU/memory remain untouched.
    This is a sensitivity operation, not a calibrated prediction of throttling.
    """

    point = operating_point
    if not point.enabled:
        return scenario
    hardware = scenario.hardware
    if hardware.metadata.get("thermal_operating_point", {}).get("enabled"):
        raise ValueError("apply a thermal operating point to the unmodified baseline, not an already derated scenario")
    affected = {component.component_id for component in hardware.components
                if component.metadata.get("thermal_domain_id") == point.domain_id}
    link_ids = {link.link_id for link in hardware.links
                if link.metadata.get("thermal_domain_id") == point.domain_id}
    if not affected and not link_ids:
        raise ValueError("unknown thermal domain: " + point.domain_id)
    ports = {(component_id, port_id)
             for link in hardware.links if link.link_id in link_ids
             for component_id, port_id in ((link.source_component, link.source_port), (link.target_component, link.target_port))}
    profiles = {kind: dict(registry) for kind, registry in scenario.component_profiles.items()}
    components = []
    for component in hardware.components:
        updates = {}
        if component.component_id in affected:
            kind = scenario.component_profile_kind(component)
            if kind is not None:
                profile_id = "{}.thermal.{}".format(component.cost_profile_id, component.component_id)
                if profile_id in profiles[kind]:
                    raise ValueError("thermal profile id already exists: " + profile_id)
                profiles[kind][profile_id] = _derate_profile(scenario.resolve_component_profile(component), point)
                updates["cost_profile_id"] = profile_id
            metadata = dict(component.metadata)
            for name in ("read_latency_ns", "write_latency_ns"):
                if name in metadata:
                    metadata[name] *= point.latency_scale
            metadata["thermal_derating"] = {"domain_id": point.domain_id, "mode": "static_operating_point", "evidence": point.evidence}
            updates.update(peak_ops_per_s=component.peak_ops_per_s * point.frequency_scale,
                           read_bandwidth_gbps=component.read_bandwidth_gbps * point.memory_bandwidth_scale,
                           write_bandwidth_gbps=component.write_bandwidth_gbps * point.memory_bandwidth_scale,
                           metadata=metadata)
        if any((component.component_id, port.port_id) in ports for port in component.ports):
            updates["ports"] = tuple(
                replace(port, bandwidth_gbps=port.bandwidth_gbps * point.link_bandwidth_scale)
                if (component.component_id, port.port_id) in ports else port
                for port in component.ports
            )
        components.append(replace(component, **updates) if updates else component)
    links = tuple(
        replace(link, bandwidth_gbps=link.bandwidth_gbps * point.link_bandwidth_scale,
                latency_ns=link.latency_ns * point.latency_scale)
        if link.link_id in link_ids else link
        for link in hardware.links
    )
    hardware = replace(hardware, components=tuple(components), links=links,
                       metadata={**hardware.metadata, "thermal_operating_point": {
                           **asdict(point), "mode": "static_derating_only", "calibrated": False,
                           "dynamic_temperature_model": False,
                       }})
    return replace(scenario, hardware=hardware, component_profiles=profiles,
                   assumptions=scenario.assumptions + ("Static thermal derating only: " + point.evidence,))
