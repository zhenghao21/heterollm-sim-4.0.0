"""Small native continuous-refill toy shared by native fast-mode tests."""
from dataclasses import replace
from pathlib import Path
import sys


def _analytical_toy_hardware(scenario, gpu_ghz):
    """Small DRAM devices on a scalar analytical no-refresh controller and a slow GPU."""
    from heterollm_sim.memory_types import DramConfig
    from heterollm_sim.physical_contract import parse_execution_physical_config

    components = []
    for component in scenario.hardware.components:
        raw = component.metadata.get("physical_memory_config")
        if raw is not None and not isinstance(raw, DramConfig):
            raw = parse_execution_physical_config(raw)
        if isinstance(raw, DramConfig):
            component = replace(component, capacity_bytes=8*1024**2, metadata={**component.metadata,
                "physical_memory_config": replace(raw, control_mode="analytical",
                    refresh_policy="disabled", channels=1, pseudo_channels_per_channel=1,
                    subchannels_per_channel=1, stacks=1, dies_per_stack=1, ranks_per_channel=1,
                    bank_groups_per_rank=1, banks_per_group=1, rows_per_bank=8192,
                    data_lanes=1, row_bytes=1024, capacity_bytes=8*1024**2,
                    metadata={**raw.metadata,
                        "control_status": "synthetic_analytical",
                        "geometry_status": "explicit_assumption"})})
        components.append(component)
    return replace(scenario, hardware=replace(scenario.hardware, components=tuple(components)),
        component_profiles={**scenario.component_profiles, "gpu": {
            key: replace(profile, tensor_core=replace(profile.tensor_core, frequency_ghz=gpu_ghz))
            for key, profile in scenario.component_profiles["gpu"].items()}})


def toy_scenario(layer_count=2, gpu_ghz=.00001, *, batch_size=1, output=3, warmup=2, measurement=4):
    """Small native HBM4 PD scenario on a scalar analytical controller.

    Both sides share a declared analytical no-refresh synthetic controller;
    independent REF and aliased-resource tests retain those guards in
    production models. Neither side is a 405B/device speed claim.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from exploring_hbf_native_config import build_native_comparison_scenario
    from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs

    layer_bytes = (32*(32+16+16+32)+3*32*64)*2
    layers = tuple(LayerSpec("dense"+str(i), "dense", 32, 64, 4,
        kv_heads=2, dtype="bf16", weight_bytes=layer_bytes) for i in range(layer_count))
    model = ModelSpec(name="native-layer-budget-toy", graph=build_model_graph_from_layer_specs(
        "native-layer-budget-toy", layers, vocabulary_size=64, max_sequence_length=256,
        embedding_weight_bytes=32*64*2, output_weight_bytes=32*64*2,
        output_head_dtype="bf16"))
    scenario = build_native_comparison_scenario("llama3_405b", "HBM4", "SHORT", tp=2,
        batch_size=batch_size, model_override=model, prompt_override=20, output_override=output,
        warmup_override=warmup, measurement_override=measurement)
    return _analytical_toy_hardware(scenario, gpu_ghz)


def _moe_toy_model(layer_count):
    from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs

    h, inter, layers = 32, 64, []
    for index in range(layer_count):
        moe = index % 2 == 1
        ffn = 3 * h * inter * (5 if moe else 1) + (h * 4 if moe else 0)
        layers.append(LayerSpec(f"layer-{index:03d}", "moe" if moe else "dense", h, inter, 4,
            kv_heads=2, dtype="bf16", weight_bytes=(32 * (32 + 16 + 16 + 32) + ffn) * 2,
            num_experts=4 if moe else 1, experts_per_token=1,
            shared_expert_intermediate_size=inter if moe else 0,
            metadata={"moe_routing": {"gating": "sigmoid", "normalize_selected_weights": False,
                "weight_scale": 1.0, "weight_application": "input", "selection_before_gating": True,
                "load_model": "deterministic_balanced_histogram"}} if moe else {}))
    return ModelSpec(name="native-toy-moe", graph=build_model_graph_from_layer_specs(
        "native-toy-moe", layers, vocabulary_size=64, max_sequence_length=256,
        embedding_weight_bytes=32 * 64 * 2, output_weight_bytes=32 * 64 * 2, output_head_dtype="bf16"))


def moe_toy_scenario(layer_count=8, *, batch_size=4, output=6, warmup=5, measurement=10, ep=1,
                     gpu_ghz=.00001):
    """Alternating dense/MoE toy on the analytical toy hardware, with EP attention groups."""
    if batch_size % ep:
        raise ValueError("batch_size must be divisible by ep")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from exploring_hbf_native_config import build_native_comparison_scenario

    scenario = build_native_comparison_scenario("llama3_405b", "HBM4", "SHORT", tp=2, ep=ep,
        batch_size=batch_size, model_override=_moe_toy_model(layer_count), prompt_override=20,
        output_override=output, warmup_override=warmup, measurement_override=measurement)
    return _analytical_toy_hardware(scenario, gpu_ghz)


def prefill_toy_scenario(layer_count=8, *, prompt=20, ep=1, full_pass=False, gpu_ghz=.00001):
    """Prefill replica of the dense/MoE toy on the analytical toy hardware."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from exploring_hbf_native_config import build_native_prefill_scenario

    scenario = build_native_prefill_scenario("llama3_405b", "SHORT", tp=2, ep=ep, full_pass=full_pass,
        model_override=_moe_toy_model(layer_count), prompt_override=prompt)
    return _analytical_toy_hardware(scenario, gpu_ghz)


def maverick_pattern_scenario(layer_count=8, *, batch_size=1, output=3, warmup=2, measurement=4,
                              tp=2, ep=1):
    """Maverick layer pattern (dense/MoE, chunked/full) on the native HBF deployment.

    ``batch_size`` is the per-replica batch and must split evenly over the ``ep``
    attention data-parallel groups.
    """
    if batch_size % ep:
        raise ValueError("batch_size must be divisible by ep")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from exploring_hbf_native_config import build_native_comparison_scenario
    from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs

    h, layers = 64, []
    for index in range(layer_count):
        moe, full = (index + 1) % 2 == 0, (index + 1) % 4 == 0
        inter = 64 if moe else 128
        ffn = 3 * h * inter * (5 if moe else 1) + (h * 4 if moe else 0)
        layers.append(LayerSpec(f"layer-{index:03d}", "moe" if moe else "dense", h, inter, 4,
            kv_heads=2, attention_head_dim=16,
            sequence_mixer="full_attention" if full else "chunked_attention",
            attention_chunk_size=0 if full else 8192, use_rope=not full,
            num_experts=4 if moe else 1, experts_per_token=1,
            shared_expert_intermediate_size=inter if moe else 0, dtype="bf16",
            weight_bytes=2 * (2 * h * h + 2 * h * 32 + ffn),
            metadata={"moe_routing": {"gating": "sigmoid", "normalize_selected_weights": False,
                "weight_scale": 1.0, "weight_application": "input", "selection_before_gating": True,
                "load_model": "deterministic_balanced_histogram"}} if moe else {}))
    model = ModelSpec(name="native-maverick-layer-windows", graph=build_model_graph_from_layer_specs(
        "native-maverick-layer-windows", layers, architecture="llama4", vocabulary_size=128,
        max_sequence_length=16384, embedding_weight_bytes=h * 128 * 2,
        output_weight_bytes=h * 128 * 2, output_head_dtype="bf16"))
    return build_native_comparison_scenario("llama4_maverick", "HBF", "SHORT", tp=tp, ep=ep,
        batch_size=batch_size, model_override=model, prompt_override=20, output_override=output,
        warmup_override=warmup, measurement_override=measurement)


def stationary_and_full_window(scenario, calibration, holdout):
    """Fast stationary forecast and the full native window's observed metrics."""
    import statistics
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig

    fast = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(calibration_layers=calibration, holdout_layers=holdout))),
        retention_policy="aggregate")
    full = run_scenario(scenario, retention_policy="aggregate").serving
    window = full.decode_measurement
    (output,) = {row.output_tokens for row in scenario.workload.requests if not row.metadata["warmup_seed"]}
    imports = {event.request_id: event.timestamp_ns for event in full.events
               if event.event_type == "external_kv_import_start"}
    seeds = {row.request_id for row in scenario.workload.requests if row.metadata["warmup_seed"]}
    tokens = {}
    for event in full.events:
        if event.event_type == "tokens_committed":
            tokens.setdefault(event.request_id, []).extend([event.timestamp_ns] * event.details["visible_tokens"])
    complete = {key: times for key, times in tokens.items() if key not in seeds and len(times) == output
                and window["start_ns"] < times[0] <= times[-1] <= window["end_ns"]}
    observed = {"mean_complete_request_tpot_ns": statistics.mean(
                    (times[-1] - times[0]) / (output - 1) for times in complete.values()),
                "cluster_tokens_per_second": window["visible_output_tokens"] * 1e9
                    / (window["end_ns"] - window["start_ns"]),
                "mean_complete_request_ingress_ttft_ns": statistics.mean(
                    times[0] - imports[key] for key, times in complete.items())}
    return fast, window, observed, len(complete)

