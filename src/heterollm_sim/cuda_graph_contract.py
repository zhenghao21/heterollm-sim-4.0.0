"""Explicit compilation inputs shared by the structural producer and runtime.

This is an applicability contract, not a latency calibration or a cache hash.
The producer writes it with its structures; changing a scenario cannot relabel
an existing structural file as belonging to another model or configuration.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path

from .cuda_graph_lifecycle import SOURCE_REVISION
from .serde import to_primitive


CONFIG_KEY = "cuda_graph_structural_program"
CONTRACT_SCHEMA = "heterollm.cuda-graph-compilation-contract/v1"
INITIALIZATION_STAGES = frozenset({"model_warmup", "model_seq_rm_probe"})
REQUEST_STAGES = INITIALIZATION_STAGES | {"request"}


def _gpu_profiles(scenario):
    for component in scenario.hardware.components:
        if component.normalized_kind == "gpu":
            profile = scenario.component_profiles["gpu"][component.cost_profile_id]
            yield component.component_id, profile.kernel_model


def validate_cuda_graph_mode(scenario, enabled):
    profiles = tuple(_gpu_profiles(scenario))
    if not profiles:
        raise ValueError("CUDA structural program requires a GPU profile")
    for component_id, profile in profiles:
        if profile is None or profile.graph_enabled is not enabled:
            raise ValueError(
                "CUDA source program Graph mode contradicts GPU profile " + component_id)


def validate_request_stages(scenario, program):
    requests = tuple(scenario.workload.requests)
    prefixes, stages = program.get("request_prefixes"), program.get("request_stages")
    ids = {request.request_id for request in requests}
    if not requests or not isinstance(prefixes, Mapping) or set(prefixes) != ids:
        raise ValueError("CUDA source program must bind every explicit request ID")
    if not isinstance(stages, Mapping) or set(stages) != ids:
        raise ValueError("CUDA source program requires an explicit stage for every request")
    if any(stage not in REQUEST_STAGES for stage in stages.values()):
        raise ValueError("CUDA source program has an unsupported request stage")
    if any(not isinstance(prefix, str) or not prefix for prefix in prefixes.values()):
        raise ValueError("CUDA source program requires non-empty request labels")
    if len(set(prefixes.values())) != len(prefixes):
        raise ValueError("CUDA source program request labels must be unique")
    seen, ordinary_seen = set(), False
    for request in sorted(requests, key=lambda item: (item.arrival_ns, item.request_id)):
        stage = stages[request.request_id]
        if stage == "request":
            ordinary_seen = True
            if prefixes[request.request_id] in INITIALIZATION_STAGES:
                raise ValueError("ordinary CUDA request cannot use an initialization label")
            continue
        if ordinary_seen or stage in seen:
            raise ValueError("CUDA initialization stages must occur once before serving requests")
        if prefixes[request.request_id] != stage:
            raise ValueError("CUDA initialization label contradicts its explicit stage")
        if request.prompt_tokens != 2 or request.output_tokens != 1:
            raise ValueError("CUDA initialization requires the compiled two-token, no-sampling call")
        if stage == "model_warmup" and not scenario.llama_cpp_config.warmup:
            raise ValueError("CUDA model warmup contradicts disabled runtime warmup")
        seen.add(stage)
    return dict(stages)


def comparable_cuda_graph_model(model):
    """Canonicalize display-only differences without changing graph semantics.

    The browser sorts the tensor directory by ID and stores canvas state in
    attributes.ui. Tensor references use IDs, not directory positions. Keep
    operator ordering, every tensor descriptor and all non-UI attributes.
    """
    result = deepcopy(model)
    if not isinstance(result, dict):
        return result
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("ui", None)
        metadata.pop("artifact_id", None)
    graph = result.get("graph")
    if not isinstance(graph, dict):
        return result
    attributes = graph.get("attributes")
    if isinstance(attributes, dict):
        attributes.pop("ui", None)
    graph.setdefault("source_operators", [])
    graph.setdefault("sub_operators", [])
    tensors = graph.get("tensors")
    if isinstance(tensors, list) and all(isinstance(tensor, dict)
            and isinstance(tensor.get("tensor_id"), str) for tensor in tensors):
        graph["tensors"] = sorted(tensors, key=lambda tensor: tensor["tensor_id"])
    return result


def build_cuda_graph_contract(scenario):
    """Describe normalized, timing-free inputs before structural compilation.

    Graph on/off and independently measured runtime costs are intentionally
    separate: both modes execute the same authored physical model. The mode
    is checked against the hardware profile when the program is admitted.
    """
    program = scenario.workload.metadata.get(CONFIG_KEY)
    if not isinstance(program, Mapping):
        raise ValueError("CUDA compilation contract requires a structural program configuration")
    # The frontend exports authoring inputs and restores runtime placement on
    # import. Bind the same realized mapping on both sides, including every
    # operator/tensor target; comparing empty authoring maps with populated
    # runtime maps would reject an unchanged scenario after /normalize.
    from .llama_scenario import prepare_llama_scenario
    scenario = prepare_llama_scenario(scenario)
    stages = validate_request_stages(scenario, program)
    workload = to_primitive(scenario.workload)
    # These fields describe the experiment or its output location, never graph
    # construction. Excluding the program also avoids a recursive contract.
    for key in (CONFIG_KEY, "cuda_graph_experiment", "cuda_graph_comparison_request_id",
                "native_output_contract_evidence", "workload_preset_id",
                "workload_preset_source", "workload_preset_source_scenario"):
        workload["metadata"].pop(key, None)
    workload.pop("name", None)
    model = comparable_cuda_graph_model(to_primitive(scenario.model))
    placement = to_primitive(scenario.placement)
    # Mapping validity bookkeeping changes after normalization; the actual
    # operators, tensors, placement, KV policy and runtime mapping remain.
    if isinstance(placement.get("metadata"), dict):
        placement["metadata"].pop("control_plane", None)
        placement["metadata"].pop("ui", None)
    hardware = to_primitive(scenario.hardware)
    if isinstance(hardware.get("metadata"), dict):
        hardware["metadata"].pop("topology_view", None)
    identities = []
    for component_id, profile in _gpu_profiles(scenario):
        if profile is None:
            raise ValueError("CUDA compilation contract requires a GPU kernel model")
        identities.append({"component_id": component_id, **{
            key: getattr(profile, key) for key in ("hardware_id", "runtime_id", "architecture")}})
    return {"schema": CONTRACT_SCHEMA, "source_revision": SOURCE_REVISION,
            "model": model, "llama_cpp_config": to_primitive(scenario.llama_cpp_config),
            "placement": placement, "workload": workload, "hardware": hardware,
            "gpu_runtime_identities": identities,
            "request_prefixes": dict(program["request_prefixes"]), "request_stages": stages}


def _first_mismatch(expected, actual, path="contract"):
    # JSON.stringify writes 0.0 as 0. Preserve numeric value across that
    # transport while keeping booleans distinct from the numbers 0 and 1.
    if type(expected) in (int, float) and type(actual) in (int, float):
        return None if expected == actual else path
    if type(expected) is not type(actual):
        return path
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return path + ".keys"
        for key in expected:
            mismatch = _first_mismatch(expected[key], actual[key], path + "." + key)
            if mismatch:
                return mismatch
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return path + ".length"
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = _first_mismatch(left, right, path + "[" + str(index) + "]")
            if mismatch:
                return mismatch
    elif expected != actual:
        return path
    return None


def require_matching_cuda_graph_contract(expected, actual, *, source):
    # Apply the same semantic view to existing producer records. This allows
    # display-only normalization without rewriting any structural evidence.
    def comparable(value):
        if not isinstance(value, dict):
            return value
        result = dict(value)
        if "model" in result:
            result["model"] = comparable_cuda_graph_model(result["model"])
        for field, ignored in (("placement", "ui"), ("hardware", "topology_view")):
            section = result.get(field)
            if isinstance(section, dict) and isinstance(section.get("metadata"), dict):
                result[field] = {**section, "metadata": {
                    key: item for key, item in section["metadata"].items() if key != ignored}}
        return result
    expected, actual = comparable(expected), comparable(actual)
    mismatch = _first_mismatch(expected, actual)
    if mismatch:
        raise ValueError("CUDA compilation contract mismatch in " + source + " at " + mismatch
                         + "; rebuild structures for the selected model and configuration")


def validate_cuda_graph_scenario_contract(scenario, *, allow_placement_probe=False):
    """Cheap frontend admission: compare authored inputs without reading traces."""
    if (allow_placement_probe
            and scenario.workload.metadata.get("_control_plane_placement_validation") is True
            and not scenario.workload.requests and scenario.workload.request_count == 1
            and scenario.workload.prompt_tokens == 1 and scenario.workload.output_tokens == 0):
        # The placement planner deliberately substitutes a one-token workload
        # with no requests. It does not admit or execute a serving program.
        # Runtime admission never enables this exemption.
        return None
    program = scenario.workload.metadata.get(CONFIG_KEY)
    if program is None:
        if any(profile is not None and getattr(profile, "runtime_calibration", None) is not None
               for _, profile in _gpu_profiles(scenario)):
            raise ValueError("independent CUDA runtime costs require a structural program in both Graph modes")
        return None
    if not isinstance(program, dict) or program.get("schema") != "heterollm.cuda-graph-source-program/v1":
        raise ValueError("invalid CUDA source-assisted structural program contract")
    if scenario.llama_cpp_config is None or scenario.llama_cpp_config.parallel != 1 or scenario.workload.mtp is not None:
        raise ValueError("CUDA source-assisted adapter covers llama single-sequence non-MTP only")
    if type(program.get("graph_enabled")) is not bool:
        raise ValueError("CUDA source program requires explicit Graph on/off mode")
    validate_cuda_graph_mode(scenario, program["graph_enabled"])
    expected = build_cuda_graph_contract(scenario)
    require_matching_cuda_graph_contract(expected, program.get("contract"), source="scenario")
    for key in ("dry_program_path", "capture_program_path"):
        if not isinstance(program.get(key), str) or not program[key]:
            raise ValueError("CUDA source program requires " + key)
    return expected


def require_cuda_graph_source_contract(path, expected):
    """Require the producer's first record, before consuming any structures."""
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                break
        else:
            raise ValueError("CUDA structural file is empty")
    if not isinstance(record, dict) or record.get("kind") != "source_contract":
        raise ValueError("CUDA structural file has no producer compilation contract: " + str(path))
    require_matching_cuda_graph_contract(expected, record.get("contract"), source=str(path))
