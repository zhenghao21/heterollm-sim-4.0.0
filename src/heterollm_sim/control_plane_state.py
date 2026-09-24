"""Stable runtime control-plane placement fingerprints.

The fingerprint describes inputs that determine whether a placement is still
valid.  Runtime workload policy is deliberately excluded: request traces,
scheduler knobs, and workload MTP policy can be re-simulated on an existing
deployment.  Hardware/model structure, calibrated profiles, residency,
parallel/KV placement, placement policy, and mapping locks remain deployment
inputs and therefore invalidate an old control-plane decision when changed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from .config import ScenarioConfig
from .serde import stable_hash, to_primitive


MAPPING_FINGERPRINT_ALGORITHM = "sha256"
MAPPING_FINGERPRINT_SCHEMA = "runtime-control-plane-v4"


_CONTROL_PLANE_SECTIONS = frozenset(("policy", "decision", "evidence"))


def _control_plane_metadata(scenario: ScenarioConfig) -> Mapping[str, Any]:
    raw = scenario.placement.metadata.get("control_plane")
    return raw if isinstance(raw, Mapping) else {}


def control_plane_section(
    scenario: ScenarioConfig, section: str
) -> Mapping[str, Any]:
    """Return one V4 ``placement.metadata.control_plane`` section.

    The normal V4 runtime never interprets legacy placement metadata.  V3
    artifacts must pass through the explicit importer before reaching this
    accessor.
    """

    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    normalized = str(section).strip()
    if normalized not in _CONTROL_PLANE_SECTIONS:
        raise ValueError(
            "control-plane section must be one of: {}".format(
                ", ".join(sorted(_CONTROL_PLANE_SECTIONS))
            )
        )
    control_plane = _control_plane_metadata(scenario)
    raw = control_plane.get(normalized, {})
    if isinstance(raw, Mapping):
        return raw
    return {}


def control_plane_policy(scenario: ScenarioConfig) -> Mapping[str, Any]:
    """Return V4 control-plane placement policy inputs."""

    return control_plane_section(scenario, "policy")


def control_plane_decision(scenario: ScenarioConfig) -> Mapping[str, Any]:
    """Return V4 control-plane placement decision outputs."""

    return control_plane_section(scenario, "decision")


def control_plane_evidence(scenario: ScenarioConfig) -> Mapping[str, Any]:
    """Return V4 control-plane decision evidence."""

    return control_plane_section(scenario, "evidence")


def _string_list(value: Any) -> tuple:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    return tuple(
        sorted(
            {
                str(item).strip()
                for item in value
                if isinstance(item, str) and str(item).strip()
            }
        )
    )


def is_control_plane_generated_tensor(
    scenario: ScenarioConfig, tensor_id: str
) -> bool:
    """Return whether ``tensor_id`` is replaceable control-plane output.

    Runtime state tensors emitted by the control plane are workload estimates,
    not user-declared capacity limits.  Locked tensors remain explicit inputs
    even if the generated-output ledger also lists them.
    """

    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    normalized = str(tensor_id).strip()
    if not normalized:
        return False
    decision = control_plane_decision(scenario)
    policy = control_plane_policy(scenario)
    generated = set(_string_list(decision.get("generated_tensor_ids", ())))
    derived_raw = decision.get("derived_tensor_bytes", {})
    derived = (
        {str(item).strip() for item in derived_raw if str(item).strip()}
        if isinstance(derived_raw, Mapping)
        else set()
    )
    locked = set(_string_list(policy.get("locked_tensor_ids", ())))
    return normalized in (generated | derived) and normalized not in locked


def _canonical_control_plane_options(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize JSON-equivalent solver options before fingerprinting.

    JavaScript serializes an integral JSON number such as ``60.0`` as ``60``.
    Mapping options cross that boundary when the Web UI stores a completed
    placement, so their fingerprint projection must not distinguish those two
    representations.  Coercing the one float-valued option back to ``float``
    also keeps the projection stable across Python and JSON runtimes.
    """

    options = dict(value)
    time_limit = options.get("time_limit_s")
    if isinstance(time_limit, (int, float)) and not isinstance(time_limit, bool):
        try:
            normalized = float(time_limit)
        except OverflowError:
            pass
        else:
            if math.isfinite(normalized):
                options["time_limit_s"] = normalized
    return options


def mapping_input_payload(
    scenario: ScenarioConfig,
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the canonical, placement-output-free control-plane projection."""

    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    policy = control_plane_policy(scenario)
    option_values: Any = options
    if option_values is None:
        stored_options = policy.get("options", {})
        option_values = stored_options if isinstance(stored_options, Mapping) else {}

    locked_op_keys = _string_list(policy.get("locked_op_keys", ()))
    locked_tensor_ids = _string_list(policy.get("locked_tensor_ids", ()))
    locked_ops = {
        key: scenario.placement.op_to_component.get(key)
        for key in locked_op_keys
    }
    locked_tensors = {
        tensor_id: {
            "component_id": scenario.placement.tensor_to_component.get(tensor_id),
            "bytes": scenario.placement.tensor_bytes.get(tensor_id),
        }
        for tensor_id in locked_tensor_ids
    }
    fixed_state_components = {
        tensor_id: scenario.placement.tensor_to_component.get(tensor_id)
        for tensor_id in (
            "kv_cache",
            "kv_cache_offload",
            "linear_state",
            "linear_state_offload",
        )
        if scenario.placement.tensor_to_component.get(tensor_id) is not None
    }
    kv_policy = to_primitive(scenario.placement.kv_policy)

    hardware = to_primitive(scenario.hardware)
    hardware_metadata = hardware.get("metadata", {})
    if isinstance(hardware_metadata, dict):
        # Editor-only coordinates/groups/viewport do not affect placement or
        # simulation.  They are preserved by the control plane, but excluded
        # from the
        # semantic staleness contract.
        hardware_metadata = dict(hardware_metadata)
        hardware_metadata.pop("topology_view", None)
        hardware["metadata"] = hardware_metadata

    model = to_primitive(scenario.model)
    graph = model.get("graph")
    if isinstance(graph, dict):
        graph = dict(graph)
        graph_attributes = graph.get("attributes", {})
        if isinstance(graph_attributes, dict):
            # Node positions, zoom, and collapsed state are editor-only.  Port,
            # tensor, transform, and component parameter edits remain semantic
            # and therefore continue to invalidate a control-plane decision.
            graph_attributes = dict(graph_attributes)
            graph_attributes.pop("ui", None)
            graph["attributes"] = graph_attributes
        model["graph"] = graph

    return {
        "fingerprint_schema": MAPPING_FINGERPRINT_SCHEMA,
        "hardware": hardware,
        "model": model,
        "profiles": {
            "components": to_primitive(scenario.component_profiles),
            "llama_cpp": to_primitive(scenario.llama_cpp_config),
            "host_orchestration": to_primitive(
                scenario.host_orchestration_profile
            ),
            "fusion": to_primitive(scenario.fusion_policy),
            "cim_interconnect": to_primitive(scenario.cim_interconnect),
        },
        "weights_resident": scenario.weights_resident,
        "placement_inputs": {
            "model_name": scenario.placement.model_name,
            "hardware_name": scenario.placement.hardware_name,
            "parallel": to_primitive(scenario.placement.parallel),
            "kv_policy": kv_policy,
            "linear_state_offload_mode": scenario.placement.metadata.get(
                "linear_state_offload_mode", "mirror"
            ),
            "aggregate_weight_backing": (
                scenario.placement.tensor_to_component.get("model_weights")
            ),
            "fixed_state_components": fixed_state_components,
            "locked_op_keys": list(locked_op_keys),
            "locked_tensor_ids": list(locked_tensor_ids),
            "locked_ops": locked_ops,
            "locked_tensors": locked_tensors,
        },
        "options": to_primitive(_canonical_control_plane_options(option_values)),
    }


def mapping_input_fingerprint(
    scenario: ScenarioConfig,
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the SHA-256 fingerprint for current control-plane inputs."""

    return stable_hash(mapping_input_payload(scenario, options=options))


def mapping_fingerprint_status(scenario: ScenarioConfig) -> Dict[str, Any]:
    """Return API-ready stored/current fingerprint and staleness fields."""

    evidence = control_plane_evidence(scenario)
    decision = control_plane_decision(scenario)
    raw_stored = evidence.get("input_fingerprint")
    stored = raw_stored.strip() if isinstance(raw_stored, str) else None
    if not stored:
        stored = None
    raw_schema = evidence.get("fingerprint_schema")
    stored_schema = (
        raw_schema.strip() if isinstance(raw_schema, str) else None
    )
    if not stored_schema:
        stored_schema = None
    current = mapping_input_fingerprint(scenario)
    # A metadata block containing only lock constraints is an input to a first
    # control-plane run.  Once generated-output metadata exists, the V4 fingerprint
    # and schema marker are mandatory and validation fails closed.
    generated_mapping = stored is not None or any(
        key in decision
        for key in (
            "generated_op_keys",
            "generated_tensor_ids",
            "surrogate",
            "derived_tensor_bytes",
            "physical_tensor_bytes",
        )
    )
    schema_matches = stored_schema == MAPPING_FINGERPRINT_SCHEMA
    return {
        "fingerprint_algorithm": MAPPING_FINGERPRINT_ALGORITHM,
        "fingerprint_schema": stored_schema,
        "current_fingerprint_schema": MAPPING_FINGERPRINT_SCHEMA,
        "input_fingerprint": stored,
        "current_input_fingerprint": current,
        "mapping_stale": generated_mapping and (
            stored is None or not schema_matches or stored != current
        ),
        "fingerprint_present": stored is not None,
    }


__all__ = [
    "MAPPING_FINGERPRINT_ALGORITHM",
    "MAPPING_FINGERPRINT_SCHEMA",
    "control_plane_decision",
    "control_plane_evidence",
    "control_plane_policy",
    "control_plane_section",
    "is_control_plane_generated_tensor",
    "mapping_fingerprint_status",
    "mapping_input_fingerprint",
    "mapping_input_payload",
]
