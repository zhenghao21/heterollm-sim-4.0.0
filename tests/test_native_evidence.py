from pathlib import Path
from types import SimpleNamespace
import pytest

from tools.native_llama_compare import (
    metric_snapshot,
    parse_perf_log,
    post_stream_json,
    post_parallel_json,
    post_parallel_stream_json,
    _aggregate_request_records,
    _native_request_record,
    _engine_boundary_timing,
    _simulator_request_timing,
    tokenize_prompt,
    build_matching_scenario,
)
from tools.replay_simulator_from_native import _explicit_native_engine_values, _native_measurements_digest
from heterollm_sim.calibration import (
    NativeCalibrationProfile,
    apply_native_calibration,
    calibrate_cost_phase,
    load_native_calibration,
    resolve_stage_calibration,
    canonical_exact_operator_key,
    audit_exact_operator_coverage,
    stage_calibration_ns_per_instance,
    memory_calibration_ns,
    launch_calibration_ns,
    synchronize_calibration_ns,
    phase_boundary_calibration_ns,
    decode_first_invocation_extra_ns,
    request_marker_calibration_ns,
)
from heterollm_sim.contracts import ResourceDemand, TaskCategory
from heterollm_sim.cost_models import CostPhase
from heterollm_sim.reference import build_reference_scenario


def test_replay_engine_values_never_backfill_legacy_prompt_eval_or_total():
    values = _explicit_native_engine_values({
        "prompt_eval_ms": {"p50_ms": 12.0},
        "total_ms": {"p50_ms": 25.0},
    })
    assert values == {"engine_ttft_ms": None, "engine_tpot_ms": None, "engine_e2e_ms": None}
    values = _explicit_native_engine_values({
        "engine_ttft_ms": {"p50_ms": 3.0},
        "engine_tpot_ms": {"p50_ms": 2.0},
        "engine_e2e_ms": {"p50_ms": 7.0},
        "prompt_eval_ms": {"p50_ms": 99.0},
    })
    assert values == {"engine_ttft_ms": 3.0, "engine_tpot_ms": 2.0, "engine_e2e_ms": 7.0}


def test_parse_perf_log_keeps_prompt_decode_total_and_graph_counts(tmp_path: Path):
    path = tmp_path / "llama.log"
    path.write_text(
        "slot print_timing: prompt eval time = 5.25 ms / 8 tokens\n"
        "slot print_timing: eval time = 18.50 ms / 4 tokens\n"
        "slot print_timing: total time = 23.75 ms / 12 tokens\n"
        "slot print_timing: graphs reused = 3\n",
        encoding="utf-8",
    )
    parsed = parse_perf_log(path)
    assert parsed["prompt"] == [{"ms": 5.25, "tokens": 8}]
    assert parsed["decode"] == [{"ms": 18.5, "tokens": 4}]
    assert parsed["total"] == [{"ms": 23.75, "tokens": 12}]
    assert parsed["graphs_reused"] == [3]


def test_metric_snapshot_ignores_comments_and_labels():
    parsed = metric_snapshot(
        "# HELP llamacpp:prompt_seconds_total x\n"
        "llamacpp:prompt_seconds_total 0.25\n"
        "llamacpp:n_busy_slots_per_decode{label=\"x\"} 1\n"
    )
    assert parsed == {"llamacpp:prompt_seconds_total": 0.25}


def test_post_stream_json_records_request_first_token_boundary(monkeypatch):
    class _Response:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def __iter__(self):
            yield b'data: {"content":"a","stop":false}\n'
            yield b'data: {"content":"b","stop":true,"timings":{"prompt_n":2,"predicted_n":2}}\n'
            yield b'data: [DONE]\n'

    monkeypatch.setattr("tools.native_llama_compare.urlopen", lambda *args, **kwargs: _Response())
    native, boundary = post_stream_json("http://127.0.0.1/completion", {"stream": True})
    assert native["timings"]["predicted_n"] == 2
    assert boundary["status"] == "measured"
    assert boundary["mode"] == "sse"
    assert boundary["first_token_source"] == "stream_first_nonempty_content"
    assert boundary["request_to_first_token_ms"] is not None
    assert boundary["request_to_end_ms"] >= boundary["request_to_first_token_ms"]


def test_post_stream_json_token_ids_prevent_empty_token_being_skipped(monkeypatch):
    class _Response:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def __iter__(self):
            yield b'data: {"content":"","tokens":[1],"stop":false}\n'
            yield b'data: {"content":"hello","tokens":[2],"stop":false}\n'
            yield b'data: {"content":"","tokens":[],"stop":true,"timings":{"predicted_n":2}}\n'

    monkeypatch.setattr("tools.native_llama_compare.urlopen", lambda *args, **kwargs: _Response())
    ticks = iter([1.0, 1.001, 1.003, 1.004, 1.005])
    monkeypatch.setattr("tools.native_llama_compare.time.perf_counter", lambda: next(ticks))
    native, boundary = post_stream_json("http://127.0.0.1/completion", {"stream": True})
    assert native["content"] == "hello"
    assert boundary["first_token_source"] == "stream_first_token_ids"
    assert boundary["request_to_first_token_ms"] < boundary["first_content_ms"]
    assert len(boundary["token_chunk_times_ms"]) == 2


def test_post_stream_json_excludes_done_control_time_from_e2e(monkeypatch):
    class _Response:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def __iter__(self):
            yield b'data: {"content":"a","tokens":[1],"stop":false}\n'
            yield b'data: {"content":"b","tokens":[2],"stop":true,"timings":{"predicted_n":2}}\n'
            yield b'data: [DONE]\n'

    monkeypatch.setattr("tools.native_llama_compare.urlopen", lambda *args, **kwargs: _Response())
    ticks = iter([1.000, 1.010, 1.020, 1.030])
    monkeypatch.setattr("tools.native_llama_compare.time.perf_counter", lambda: next(ticks))
    _native, boundary = post_stream_json("http://127.0.0.1/completion", {"stream": True})
    assert boundary["request_to_end_ms"] == pytest.approx(20.0)
    assert boundary["stream_end_ms"] == pytest.approx(30.0)
    assert boundary["stream_control_overhead_ms"] == pytest.approx(10.0)


def test_native_request_record_separates_client_and_engine_tpot():
    record = _native_request_record(
        {"timings": {"prompt_ms": 4.0, "predicted_ms": 30.0, "prompt_n": 8, "predicted_n": 3}},
        {"request_to_first_token_ms": 10.0, "request_to_end_ms": 30.0, "status": "measured"},
        0, 8,
    )
    assert record["tpot_ms"] == 10.0
    assert record["client_tpot_ms"] == 10.0
    assert record["engine_tpot_ms"] == 15.0


def test_explicit_engine_boundary_has_precedence_over_client_stream():
    boundary = {
        "status": "measured",
        "request_to_first_token_ms": 100.0,
        "request_to_end_ms": 140.0,
        "engine_boundary": {
            "status": "measured",
            "request_begin_ns": 1_000_000,
            "first_token_ns": 4_000_000,
            "last_token_ns": 9_000_000,
            "token_times_ns": [4_000_000, 9_000_000],
        },
    }
    result = _native_request_record(
        {"timings": {"prompt_ms": 99.0, "predicted_ms": 99.0, "predicted_n": 2}},
        boundary, 0, 2,
    )
    assert result["engine_timing_status"] == "marker_proven"
    assert result["engine_ttft_ms"] == pytest.approx(3.0)
    assert result["engine_tpot_ms"] == pytest.approx(5.0)
    assert result["engine_e2e_ms"] == pytest.approx(8.0)
    assert result["client_ttft_ms"] == pytest.approx(100.0)


def test_counter_engine_timing_is_explicit_and_separate_from_client():
    result = _native_request_record(
        {"timings": {"prompt_ms": 4.0, "predicted_ms": 30.0, "predicted_n": 3}},
        {"request_to_first_token_ms": 10.0, "request_to_end_ms": 30.0, "status": "measured"},
        0, 8,
    )
    assert result["engine_timing_status"] == "counter_proven"
    assert result["engine_ttft_ms"] == pytest.approx(4.0)
    assert result["engine_e2e_ms"] == pytest.approx(34.0)
    assert result["client_ttft_ms"] == pytest.approx(10.0)


def test_simulator_engine_timing_uses_prefill_batch_boundary():
    simulation = SimpleNamespace(serving=SimpleNamespace(
        batches=(SimpleNamespace(kind="prefill", request_ids=("request-0000",), end_ns=6_000_000),),
        events=(SimpleNamespace(event_type="tokens_committed", request_id="request-0000", timestamp_ns=6_000_000),
                SimpleNamespace(event_type="tokens_committed", request_id="request-0000", timestamp_ns=16_000_000)),
    ))
    metric = SimpleNamespace(request_id="request-0000", arrival_ns=0.0,
                             start_ns=1_000_000, first_token_ns=6_000_000,
                             finish_ns=16_000_000, tpot_ns=2_000_000,
                             visible_output_tokens=6)
    result = _simulator_request_timing(simulation, metric)
    assert result["engine_ttft_ms"] == pytest.approx(5.0)
    assert result["engine_e2e_ms"] == pytest.approx(15.0)
    assert result["client_ttft_ms"] == pytest.approx(6.0)
    assert result["engine_ttft_source"] == "simulator.first_engine_token-start"


def test_native_request_record_tpot_is_undefined_for_single_output_token():
    record = _native_request_record(
        {"timings": {"predicted_ms": 4.0, "predicted_n": 1}},
        {"request_to_first_token_ms": 10.0, "request_to_end_ms": 10.0, "status": "measured"},
        0, 8,
    )
    assert record["client_tpot_ms"] is None
    assert record["engine_tpot_ms"] is None


def test_tokenize_prompt_uses_locked_tokenizer_before_native(monkeypatch, tmp_path: Path):
    tokenizer = tmp_path / "llama-tokenize.exe"
    tokenizer.write_bytes(b"stub")

    class _Completed:
        returncode = 0
        stdout = "[1, 2, 3]\nTotal number of tokens: 3\n"
        stderr = ""

    calls = []
    monkeypatch.setattr("tools.native_llama_compare.subprocess.run", lambda cmd, **kwargs: (calls.append(cmd) or _Completed()))
    result = tokenize_prompt("model.gguf", "Hi", tokenizer_exe=tokenizer)
    assert result["count"] == 3
    assert result["ids"] == [1, 2, 3]
    assert "--show-count" in calls[0] and "--ids" in calls[0]


def test_observed_small_prefill_chunk_does_not_leak_to_unseen_shapes():
    from heterollm_sim.config import model_from_dict
    from heterollm_sim.model_presets import materialize_model_payload

    model = model_from_dict(materialize_model_payload("qwen3_8-27b"))
    common = dict(ctx=2048, batch=64, ubatch=64, threads=16, gpu_layers=0, model=model)
    captured = build_matching_scenario(8, 8, parallel=1, **common)
    held_out = build_matching_scenario(216, 128, parallel=4, **common)
    assert captured.workload.scheduler.prefill_chunk_tokens == 4
    assert held_out.workload.scheduler.prefill_chunk_tokens == 64


def test_parallel_json_uses_all_slots_and_preserves_one_request_behavior(monkeypatch):
    calls = []

    def fake_post(url, payload):
        calls.append((url, payload))
        return {"ok": True, "index": len(calls)}

    monkeypatch.setattr("tools.native_llama_compare.post_json", fake_post)
    result = post_parallel_json("http://localhost/completion", {"stream": False}, 4)
    assert len(result) == 4
    assert len(calls) == 4
    calls.clear()
    one = post_parallel_json("http://localhost/completion", {"stream": False}, 1)
    assert len(one) == 1
    assert len(calls) == 1


def test_parallel_stream_json_keeps_one_boundary_per_slot(monkeypatch):
    calls = []

    def fake_stream(url, payload):
        calls.append(payload)
        return ({"timings": {"predicted_n": 2}}, {
            "status": "measured",
            "mode": "sse",
            "request_to_first_token_ms": 1.0,
            "request_to_end_ms": 2.0,
        })

    monkeypatch.setattr("tools.native_llama_compare.post_stream_json", fake_stream)
    result = post_parallel_stream_json("http://localhost/completion", {"stream": True}, 2)
    assert len(result) == 2
    assert len(calls) == 2
    assert all(item[1]["batch_client_makespan_ms"] == 2.0 for item in result)
    assert all(item[1]["batch_end_monotonic_s"] >= item[1]["batch_start_monotonic_s"] for item in result)


def test_request_aggregate_reports_p50_p90_and_makespan():
    records = [
        {"prompt_eval_ms": 1.0, "eval_ms": 4.0, "tpot_ms": 2.0,
         "total_ms": 5.0, "request_to_first_token_ms": 3.0,
         "request_to_end_ms": 6.0},
        {"prompt_eval_ms": 2.0, "eval_ms": 8.0, "tpot_ms": 4.0,
         "total_ms": 10.0, "request_to_first_token_ms": 5.0,
         "request_to_end_ms": 11.0},
        {"prompt_eval_ms": 3.0, "eval_ms": 12.0, "tpot_ms": 6.0,
         "total_ms": 15.0, "request_to_first_token_ms": 7.0,
         "request_to_end_ms": 16.0},
    ]
    aggregate = _aggregate_request_records(records, batch_client_wall_ms=20.0)
    assert aggregate["request_count"] == 3
    assert aggregate["request_to_first_token_ms"]["p50_ms"] == 5.0
    assert aggregate["request_to_first_token_ms"]["p90_ms"] == 6.6
    assert aggregate["request_to_end_ms"]["max_ms"] == 16.0
    assert aggregate["makespan_ms"] == 16.0
    assert aggregate["batch_client_wall_ms"] == 20.0


def test_native_calibration_changes_only_gpu_frontend_profile():
    scenario = build_reference_scenario()
    profile = NativeCalibrationProfile(launch_ns_per_call=1234.0, source="test")
    calibrated = apply_native_calibration(scenario, profile, apply_launch=True)
    assert calibrated.placement.metadata["native_calibration"]["launch_ns_per_call"] == 1234.0
    for gpu in calibrated.component_profiles["gpu"].values():
        assert gpu.kernel_launch_ns == 1234.0

def test_native_calibration_defaults_to_evidence_only():
    scenario = build_reference_scenario()
    calibrated = apply_native_calibration(scenario, NativeCalibrationProfile(launch_ns_per_call=1234.0, source="test"))
    assert calibrated.component_profiles == scenario.component_profiles
    assert "native_calibration" in calibrated.placement.metadata


def _semantic_profile() -> NativeCalibrationProfile:
    return NativeCalibrationProfile(
        kernel_stage_mapping={
            "attention_qkv": {
                "status": "calibrated",
                "phases": {
                    "decode": {
                        "status": "calibrated",
                        "train_ns_per_instance": 125.0,
                        "train_instances": 4,
                        "token_shapes": {
                            "1x14x1x1": {
                                "status": "calibrated",
                                "train_ns_per_instance": 100.0,
                                "train_instances": 4,
                            }
                        },
                    }
                },
            }
        },
        synchronize_ns_per_call=7.0,
    )


def test_stage_calibration_requires_exact_projection_and_shape():
    profile = _semantic_profile()
    phase = CostPhase(
        "gpu_gemm", TaskCategory.COMPUTE,
        (ResourceDemand("gpu0.tensor_core", 900.0), ResourceDemand("gpu0.hbm", 30.0)),
    )
    metadata = {
        "projection_id": "attention.qkv",
        "calibration_stage": "attention_qkv",
        "phase": "decode",
        "token_shape": "1x14x1x1",
    }
    calibrated = calibrate_cost_phase(phase, metadata, profile)
    assert calibrated.demands[0].service_ns == 100.0
    assert phase.demands[1].service_ns == calibrated.demands[1].service_ns
    assert calibrate_cost_phase(phase, {**metadata, "projection_id": ""}, profile) == phase
    assert calibrate_cost_phase(phase, {**metadata, "token_shape": "missing"}, profile) == phase
    assert stage_calibration_ns_per_instance(profile, stage="unknown", phase="decode") is None


def test_stage_and_launch_calibration_compose_on_one_phase():
    """Applying launch evidence must not discard an exact operator rate.

    A phase carries separate frontend and compute demands.  Both pieces are
    measured independently in llama.cpp: the operator-wall profile replaces
    the compute demand while the launch profile replaces only frontend time.
    The two calibrations therefore have to compose in one call.
    """
    profile = NativeCalibrationProfile(
        kernel_stage_mapping={
            "ffn": {
                "status": "calibrated",
                "phases": {
                    "decode": {
                        "status": "calibrated",
                        "token_shapes": {
                            "1x1": {
                                "status": "calibrated",
                                "train_ns_per_instance": 125.0,
                                "train_instances": 1,
                            }
                        },
                    }
                },
            }
        },
        decode_launch_ns_per_call=7.0,
    )
    phase = CostPhase(
        "gpu_ffn", TaskCategory.COMPUTE,
        (
            ResourceDemand("gpu0.frontend", 900.0),
            ResourceDemand("gpu0.tensor_core", 800.0),
            ResourceDemand("gpu0.hbm", 30.0),
        ),
    )
    metadata = {
        "projection_id": "mlp.down",
        "calibration_stage": "ffn",
        "phase": "decode",
        "token_shape": "1x1",
        "calibration_instances": 1,
    }
    calibrated = calibrate_cost_phase(phase, metadata, profile)
    assert calibrated.demands[0].service_ns == 7.0
    assert calibrated.demands[1].service_ns == 125.0
    assert calibrated.demands[2].service_ns == 30.0
    assert calibrated.metadata["native_calibration_applied"] is True
    assert calibrated.metadata["native_launch_calibration_applied"] is not None


def test_phase_scoped_launch_rate_does_not_fall_back_to_global():
    profile = NativeCalibrationProfile(
        launch_ns_per_call=8000.0,
        prefill_launch_ns_per_call=5000.0,
    )
    assert launch_calibration_ns(profile, "prefill") == 5000.0
    assert launch_calibration_ns(profile, "decode") is None


def test_linear_attention_aux_calibration_accepts_primitive_without_projection():
    profile = NativeCalibrationProfile(kernel_stage_mapping={
        "linear_attention_aux": {"status": "calibrated", "phases": {
            "decode": {"status": "calibrated", "token_shapes": {
                "1x1": {"status": "calibrated", "train_ns_per_instance": 42.0, "train_instances": 1}
            }}
        }}
    })
    phase = CostPhase("linear_aux", TaskCategory.COMPUTE,
                      (ResourceDemand("gpu0.tensor_core", 900.0), ResourceDemand("gpu0.hbm", 30.0)))
    metadata = {"coverage_component": "linear_attention", "linear_op": "local_conv",
                "phase": "decode", "token_shape": "1x1"}
    calibrated = calibrate_cost_phase(phase, metadata, profile)
    assert calibrated.demands[0].service_ns == 42.0
    assert calibrated.demands[1].service_ns == 30.0


def test_cpu_memory_calibration_uses_stage_bandwidth_and_keeps_bytes():
    profile = NativeCalibrationProfile(kernel_stage_mapping={
        "ffn": {"status": "calibrated", "phases": {"decode": {
            "status": "calibrated", "token_shapes": {
                "1x1": {"status": "calibrated", "train_effective_bandwidth_gbps": 40.0,
                        "train_total_bytes": 4000, "train_instances": 1,
                        "train_ns_per_instance": 100}
            }
        }}}
    }, coverage_status="covered")
    phase = CostPhase("cpu", TaskCategory.COMPUTE,
                      (ResourceDemand("cpu0.pipeline", 7.0),
                       ResourceDemand("cpu0.memory", 1000.0, bytes_moved=4000)))
    metadata = {"calibration_stage": "ffn", "phase": "decode",
                "token_shape": "1x1", "projection_id": "mlp.down"}
    assert memory_calibration_ns(phase, metadata, profile) == 100.0
    calibrated = calibrate_cost_phase(phase, metadata, profile, apply_memory=True)
    assert calibrated.demands[0].service_ns == 7.0
    assert calibrated.demands[1].service_ns == 100.0
    assert calibrated.demands[1].bytes_moved == 4000


def test_gemm_phase_token_shape_uses_operation_envelope(monkeypatch):
    """GEMM phase metadata must retain the exact N x M invocation shape.

    Quantization metadata is attached to the operation envelope assembled by
    ``_add_rank_gemm``; it is not part of each CostPhase's local metadata.
    This regression keeps shape-specific native calibration usable for a
    decode M=1 invocation (and still derives the shape when the phase itself
    has no ``gemm_n``/``gemm_m`` fields).
    """
    import heterollm_sim.planner as planner

    original = planner._workload_quantization_metadata

    def with_dimensions(scenario, workload, name, operation_metadata, **kwargs):
        value = dict(original(scenario, workload, name, operation_metadata, **kwargs))
        value.setdefault("gemm_n", workload.n)
        value.setdefault("gemm_m", workload.m)
        return value

    monkeypatch.setattr(
        planner, "_workload_quantization_metadata", with_dimensions
    )
    schedule = planner.compile_scenario(build_reference_scenario())
    task = next(
        task for task in schedule.tasks if task.name.endswith("qkv.gpu_gemm")
    )
    assert task.metadata["phase_metadata"]["token_shape"] == "1024x64x1x1"


def test_native_phase_detection_accepts_serving_cohort_prefixes():
    from heterollm_sim.planner import _execution_phase_from_name

    assert _execution_phase_from_name("cohort-000001.decode.layer-000.mlp_down") == "decode"
    assert _execution_phase_from_name("cohort-000000.prefill.group0001.layer-000") == "prefill"
    assert _execution_phase_from_name("decode0001.layer-000") == "decode"
    assert _execution_phase_from_name("decode_buffer.allocate") is None


def test_stage_identity_mismatch_and_sync_require_explicit_boundary():
    profile = _semantic_profile()
    profile = NativeCalibrationProfile(
        **{**profile.__dict__, "model_sha256": "model-a", "hardware_fingerprint": "hw-a"}
    )
    phase = CostPhase("gpu_gemm", TaskCategory.COMPUTE, (ResourceDemand("gpu0.tensor_core", 10.0),))
    metadata = {"projection_id": "p", "calibration_stage": "attention_qkv", "phase": "decode", "token_shape": "1x14x1x1"}
    assert calibrate_cost_phase(phase, metadata, profile, model_sha256="model-b", hardware_fingerprint="hw-a") == phase
    assert synchronize_calibration_ns({"event_kind": "sync"}, profile) is None
    assert synchronize_calibration_ns({"event_kind": "sync", "sync_boundary_count": 3}, profile) == 21.0


def test_phase_boundary_requires_explicit_policy_and_is_one_value_per_invocation():
    profile = NativeCalibrationProfile(
        phase_boundary_ns_per_invocation={"prefill": 2075657.0, "decode": 1442557.714},
        phase_boundary_policy="one_task_per_phase_invocation",
        coverage_status="covered",
    )
    assert phase_boundary_calibration_ns(profile, "prefill") == 2075657.0
    assert phase_boundary_calibration_ns(profile, "decode") == 1442557.714
    blocked = NativeCalibrationProfile(
        phase_boundary_ns_per_invocation={"prefill": 2075657.0},
        coverage_status="covered",
    )
    assert phase_boundary_calibration_ns(blocked, "prefill") is None


def test_decode_first_invocation_residual_is_one_time_and_identity_gated():
    profile = NativeCalibrationProfile(
        decode_first_invocation_extra_ns=8_407_471.0,
        decode_first_invocation_policy="first_decode_only",
        coverage_status="covered",
        model_sha256="model-a",
        hardware_fingerprint="hw-a",
        runtime_fingerprint="rt-a",
    )
    assert decode_first_invocation_extra_ns(
        profile,
        first_invocation=True,
        model_sha256="model-a",
        hardware_fingerprint="hw-a",
        runtime_fingerprint="rt-a",
    ) == 8_407_471.0
    assert decode_first_invocation_extra_ns(
        profile,
        first_invocation=False,
        model_sha256="model-a",
        hardware_fingerprint="hw-a",
        runtime_fingerprint="rt-a",
    ) is None
    assert decode_first_invocation_extra_ns(
        profile,
        first_invocation=True,
        model_sha256="model-b",
        hardware_fingerprint="hw-a",
        runtime_fingerprint="rt-a",
    ) is None
    assert decode_first_invocation_extra_ns(
        NativeCalibrationProfile(
            decode_first_invocation_extra_ns=8_407_471.0,
            decode_first_invocation_policy="first_decode_only",
            coverage_status="blocked",
        ),
        first_invocation=True,
    ) is None


def test_phase_boundary_lowering_adds_one_task_after_frontend_not_per_operator():
    from dataclasses import replace
    from heterollm_sim.planner import compile_scenario

    base = build_reference_scenario()
    placement = replace(base.placement, metadata={
        **base.placement.metadata,
        "native_calibration": NativeCalibrationProfile(
            phase_boundary_ns_per_invocation={"prefill": 1234.0},
            phase_boundary_policy="one_task_per_phase_invocation",
            coverage_status="covered",
        ).to_dict(),
        "native_calibration_apply_phase_boundary": True,
    })
    workload = replace(base.workload, mtp=None)
    scenario = replace(base, placement=placement, workload=workload)
    schedule = compile_scenario(scenario)
    boundaries = [task for task in schedule.tasks
                  if task.metadata.get("event_kind") == "native_phase_boundary"]
    assert len(boundaries) == 1
    assert boundaries[0].metadata["runtime_phase"] == "prefill"
    assert boundaries[0].demands[0].service_ns == 1234.0
    assert not any(task.metadata.get("event_kind") == "native_phase_boundary"
                   for task in schedule.tasks if task.name.endswith("kernel_launch"))


def _request_marker_profile(**kwargs):
    values = {
        "request_marker_ns": {
            "request_begin": 11.0,
            "first_token": 23.0,
            "request_end": 37.0,
        },
        "request_marker_policy": "additive_once_per_request",
        "request_marker_evidence": {
            "request_begin": {"status": "calibrated"},
            "first_token": {"status": "calibrated"},
            "request_end": {"status": "calibrated"},
        },
        "request_shape": {"prompt_tokens": 2, "output_tokens": 3,
                          "prompt_fingerprint": "prompt-a"},
        "coverage_status": "covered",
        "model_sha256": "model-a",
        "hardware_fingerprint": "hw-a",
        "runtime_fingerprint": "rt-a",
    }
    values.update(kwargs)
    return NativeCalibrationProfile(**values)


def test_request_marker_calibration_requires_exact_identity_shape_and_evidence():
    profile = _request_marker_profile()
    common = dict(model_sha256="model-a", hardware_fingerprint="hw-a",
                  runtime_fingerprint="rt-a", prompt_tokens=2,
                  output_tokens=3, prompt_fingerprint="prompt-a")
    assert request_marker_calibration_ns(profile, "first_token", **common) == 23.0
    assert request_marker_calibration_ns(profile, "request_done", **common) == 37.0
    assert request_marker_calibration_ns(profile, "first_token", **{**common, "output_tokens": 4}) is None
    assert request_marker_calibration_ns(profile, "first_token", **{**common, "prompt_fingerprint": "prompt-b"}) is None
    assert request_marker_calibration_ns(profile, "first_token", **{**common, "model_sha256": "model-b"}) is None
    assert request_marker_calibration_ns(profile, "unknown", **common) is None
    assert request_marker_calibration_ns(
        _request_marker_profile(request_marker_policy=None), "first_token", **common
    ) is None


def test_request_marker_lowering_is_explicit_once_per_request_and_fail_closed():
    from dataclasses import replace
    from heterollm_sim.planner import compile_scenario

    base = build_reference_scenario()
    profile = _request_marker_profile(model_sha256=None,
                                      hardware_fingerprint=None,
                                      runtime_fingerprint=None)
    placement = replace(base.placement, metadata={
        **base.placement.metadata,
        "native_calibration": profile.to_dict(),
        "native_calibration_apply_request_boundary": True,
        "prompt_fingerprint": "prompt-a",
        "hardware_fingerprint": "hw-a",
    })
    request = replace(base.workload.requests[0], prompt_tokens=2, output_tokens=3)
    scenario = replace(base, placement=placement,
                       workload=replace(base.workload, requests=(request,), mtp=None),
                       llama_cpp_config=replace(base.llama_cpp_config,
                                                threads=base.llama_cpp_config.threads)
                       if base.llama_cpp_config is not None else None)
    schedule = compile_scenario(scenario)
    boundaries = [task for task in schedule.tasks
                  if task.metadata.get("event_kind") == "native_request_marker_boundary"]
    assert [task.metadata["request_marker"] for task in boundaries] == [
        "request_begin", "first_token", "request_end"
    ]
    assert [task.demands[0].service_ns for task in boundaries] == [11.0, 23.0, 37.0]
    blocked = replace(placement, metadata={
        **placement.metadata, "prompt_fingerprint": "wrong-prompt"
    })
    blocked_schedule = compile_scenario(replace(scenario, placement=blocked))
    assert not any(task.metadata.get("event_kind") == "native_request_marker_boundary"
                   for task in blocked_schedule.tasks)


def test_load_native_semantic_profile(tmp_path):
    path = tmp_path / "semantic.json"
    path.write_text('{"schema":"native-semantic-calibration/v1","stages":{"kv":{"status":"blocked_no_semantic_evidence"}}}', encoding="utf-8")
    loaded = load_native_calibration(path)
    assert loaded.kernel_stage_mapping["kv"]["status"] == "blocked_no_semantic_evidence"


def test_load_native_semantic_profile_preserves_phase_boundary_evidence(tmp_path):
    path = tmp_path / "semantic-boundary.json"
    path.write_text(
        '{"schema":"native-semantic-calibration/v1",'
        '"coverage":{"status":"covered"},'
        '"phase_boundary_ns_per_invocation":{"prefill":11,"decode":7},'
        '"phase_boundary_policy":"one_task_per_phase_invocation",'
        '"stages":{}}', encoding="utf-8")
    loaded = load_native_calibration(path)
    assert loaded.phase_boundary_policy == "one_task_per_phase_invocation"
    assert phase_boundary_calibration_ns(loaded, "decode") == 7.0


def test_blocked_semantic_profile_is_evidence_only(tmp_path):
    path = tmp_path / "blocked.json"
    path.write_text(
        '{"schema":"native-semantic-calibration/v1",'
        '"coverage":{"status":"blocked"},'
        '"stages":{"attention_qkv":{"status":"calibrated",'
        '"phases":{"decode":{"status":"calibrated",'
        '"train_ns_per_instance":10,"train_instances":1}}}}}',
        encoding="utf-8",
    )
    loaded = load_native_calibration(path)
    assert loaded.coverage_status == "blocked"
    assert stage_calibration_ns_per_instance(loaded, stage="attention_qkv", phase="decode") is None


def test_identity_mismatch_semantic_profile_is_evidence_only(tmp_path):
    path = tmp_path / "mismatch.json"
    path.write_text(
        '{"schema":"native-semantic-calibration/v1",'
        '"train_identity":{"model_sha256":"a"},'
        '"holdout_identity":{"model_sha256":"b"},'
        '"coverage":{"status":"covered"},'
        '"stages":{"attention_qkv":{"status":"calibrated",'
        '"phases":{"decode":{"status":"calibrated",'
        '"train_ns_per_instance":10,"train_instances":1}}}}}',
        encoding="utf-8",
    )
    loaded = load_native_calibration(path)
    assert loaded.identity_mismatch is True
    assert stage_calibration_ns_per_instance(loaded, stage="attention_qkv", phase="decode") is None


def test_operator_wall_basis_prefers_wall_coefficient(tmp_path):
    path = tmp_path / "wall.json"
    path.write_text(
        '{"schema":"native-semantic-calibration/v1","calibration_basis":"operator_wall",'
        '"coverage":{"status":"covered"},"stages":{"attention_qkv":{"status":"calibrated",'
        '"phases":{"decode":{"status":"calibrated","train_ns_per_instance":2,'
        '"train_instances":1,"wall_ns_per_instance":7,"wall_train_instances":1}}}}}',
        encoding="utf-8",
    )
    loaded = load_native_calibration(path)
    assert loaded.calibration_basis == "operator_wall"
    assert stage_calibration_ns_per_instance(loaded, stage="attention_qkv", phase="decode") == 7


def test_operator_wall_basis_accepts_wall_token_shape_bucket(tmp_path):
    path = tmp_path / "wall-shape.json"
    path.write_text(
        '{"schema":"native-semantic-calibration/v1","calibration_basis":"operator_wall",'
        '"coverage":{"status":"covered"},"stages":{"ffn":{"status":"calibrated",'
        '"phases":{"decode":{"status":"calibrated","wall_token_shapes":{'
        '"17408x1x1x1":{"wall_status":"calibrated","wall_ns_per_instance":496192.5,'
        '"wall_train_instances":2}}}}}}}',
        encoding="utf-8",
    )
    loaded = load_native_calibration(path)
    assert stage_calibration_ns_per_instance(
        loaded, stage="ffn", phase="decode", token_shape="17408x1x1x1"
    ) == 496192.5


def test_kernel_shape_policy_uses_exact_bucket_and_phase_fallback():
    """Kernel shape evidence is opt-in and fails closed for a missing shape."""
    profile = NativeCalibrationProfile(
        calibration_basis="kernel",
        kernel_stage_mapping={"ffn": {"status": "calibrated", "phases": {
            "prefill": {"status": "calibrated", "train_ns_per_instance": 250.0,
                         "train_instances": 1, "token_shapes": {
                             "896x9x1x1": {"status": "calibrated",
                                           "train_ns_per_instance": 90.0,
                                           "train_instances": 1}}}
        }}}
    )
    phase = CostPhase("gpu_ffn", TaskCategory.COMPUTE,
                      (ResourceDemand("gpu0.tensor_core", 900.0),))
    base = {"projection_id": "mlp.gate", "calibration_stage": "ffn",
            "phase": "prefill", "native_calibration_shape_policy":
            "kernel_shape_if_available"}
    hit = calibrate_cost_phase(phase, {**base, "token_shape": "896x9x1x1"}, profile)
    assert hit.demands[0].service_ns == 90.0
    miss = calibrate_cost_phase(phase, {**base, "token_shape": "896x12x1x1"}, profile)
    assert miss.demands[0].service_ns == 250.0


def test_exact_operator_key_requires_all_dimensions_and_normalizes_whitespace():
    assert canonical_exact_operator_key(
        stage=" ffn ", phase="decode", token_shape="896x1x1x1",
        dtype="f32", layout="contiguous", kernel_family="gpu_mmq",
    ) == "ffn|decode|896x1x1x1|f32|contiguous|gpu_mmq"
    assert canonical_exact_operator_key(
        stage="ffn", phase="decode", token_shape="896x1x1x1",
        dtype="f32", layout=None, kernel_family="gpu_mmq",
    ) is None


def test_exact_operator_gate_hits_only_full_key_and_fails_closed_on_shape_miss():
    profile = NativeCalibrationProfile(
        calibration_basis="kernel",
        kernel_stage_mapping={"exact_operator_keys": {
            "ffn|decode|896x1x1x1|f32|contiguous|gpu_mmq": {
                "status": "calibrated", "train_ns_per_instance": 77.0,
                "train_instances": 1,
            }
        }},
    )
    records = [
        {"stage": "ffn", "phase": "decode", "semantic_shape": "896x1x1x1",
         "semantic_type": "f32", "semantic_layout": "contiguous", "kernel_family": "gpu_mmq"},
        {"stage": "ffn", "phase": "decode", "semantic_shape": "896x2x1x1",
         "semantic_type": "f32", "semantic_layout": "contiguous", "kernel_family": "gpu_mmq"},
        {"stage": "ffn", "phase": "decode", "semantic_shape": "896x1x1x1",
         "semantic_type": "f32", "kernel_family": "gpu_mmq"},
    ]
    audit = audit_exact_operator_coverage(profile, records)
    assert audit["records"] == 3
    assert audit["exact_hits"] == 1
    assert audit["fallback_records"] == 2
    assert audit["exact_hit_rate_pct"] == pytest.approx(100 / 3)
    assert audit["fallback_reasons"]["exact_operator_key_uncovered"] == 1
    assert audit["fallback_reasons"]["missing_operator_key_dimension"] == 1


def test_exact_operator_policy_does_not_charge_uncovered_shape():
    profile = NativeCalibrationProfile(
        calibration_basis="kernel",
        kernel_stage_mapping={"exact_operator_keys": {
            "ffn|decode|896x1x1x1|f32|contiguous|gpu_mmq": {
                "status": "calibrated", "train_ns_per_instance": 77.0,
                "train_instances": 1,
            }
        }},
    )
    phase = CostPhase("gpu_ffn", TaskCategory.COMPUTE,
                      (ResourceDemand("gpu0.tensor_core", 900.0),))
    base = {"projection_id": "mlp.gate", "calibration_stage": "ffn",
            "phase": "decode", "semantic_type": "f32",
            "semantic_layout": "contiguous", "kernel_family": "gpu_mmq",
            "native_calibration_operator_key_policy": "exact"}
    hit = calibrate_cost_phase(phase, {**base, "token_shape": "896x1x1x1"}, profile)
    assert hit.demands[0].service_ns == 77.0
    miss = calibrate_cost_phase(phase, {**base, "token_shape": "896x2x1x1"}, profile)
    assert miss.demands == phase.demands
    assert miss.metadata["native_calibration_operator_key_resolution"]["mode"] == "analytical_fallback"


def test_kernel_shape_policy_does_not_use_shape_for_wall_profile():
    profile = NativeCalibrationProfile(
        calibration_basis="operator_wall",
        kernel_stage_mapping={"ffn": {"status": "calibrated", "phases": {
            "decode": {"status": "calibrated", "train_ns_per_instance": 125.0,
                        "train_instances": 1, "wall_ns_per_instance": 180.0,
                        "wall_train_instances": 1, "wall_token_shapes": {
                            "896x1x1x1": {"wall_status": "calibrated",
                                           "wall_ns_per_instance": 999.0,
                                           "wall_train_instances": 1}}}
        }}}
    )
    phase = CostPhase("gpu_ffn", TaskCategory.COMPUTE,
                      (ResourceDemand("gpu0.tensor_core", 900.0),))
    metadata = {"projection_id": "mlp.down", "calibration_stage": "ffn",
                "phase": "decode", "token_shape": "896x1x1x1",
                "native_calibration_shape_policy": "kernel_shape_if_available"}
    calibrated = calibrate_cost_phase(phase, metadata, profile)
    assert calibrated.demands[0].service_ns == 180.0


def test_shape_interpolation_is_explicit_and_bounded():
    profile = NativeCalibrationProfile(
        calibration_basis="kernel",
        kernel_stage_mapping={"ffn": {"status": "calibrated", "phases": {
            "prefill": {"status": "calibrated", "token_shapes": {
                "896x2x1x1": {"status": "calibrated", "train_ns_per_instance": 100.0, "train_instances": 1, "holdout_instances": 1},
                "896x8x1x1": {"status": "calibrated", "train_ns_per_instance": 220.0, "train_instances": 1, "holdout_instances": 1},
            }}
        }}}
    )
    hit = resolve_stage_calibration(
        profile, stage="ffn", phase="prefill", token_shape="896x5x1x1",
        allow_interpolation=True,
    )
    assert hit["mode"] == "interpolated"
    assert hit["coefficient"] == 160.0
    assert hit["source_shapes"] == ["896x2x1x1", "896x8x1x1"]
    miss = resolve_stage_calibration(
        profile, stage="ffn", phase="prefill", token_shape="896x12x1x1",
        allow_interpolation=True,
    )
    assert miss["mode"] == "extrapolation"
    assert miss["coefficient"] is None


def test_interpolation_policy_keeps_extrapolated_shape_analytical():
    profile = NativeCalibrationProfile(
        calibration_basis="kernel",
        kernel_stage_mapping={"ffn": {"status": "calibrated", "phases": {
            "prefill": {"status": "calibrated", "train_ns_per_instance": 999.0,
                         "train_instances": 1, "token_shapes": {
                "896x2x1x1": {"status": "calibrated", "train_ns_per_instance": 100.0,
                               "train_instances": 1, "holdout_instances": 1},
                "896x8x1x1": {"status": "calibrated", "train_ns_per_instance": 220.0,
                               "train_instances": 1, "holdout_instances": 1}}}
        }}}
    )
    phase = CostPhase("gpu_ffn", TaskCategory.COMPUTE,
                      (ResourceDemand("gpu0.tensor_core", 900.0),))
    metadata = {"projection_id": "mlp.gate", "calibration_stage": "ffn",
                "phase": "prefill", "token_shape": "896x12x1x1",
                "native_calibration_shape_policy": "interpolate_within_evidence"}
    result = calibrate_cost_phase(phase, metadata, profile)
    assert result.demands == phase.demands
    assert result.metadata["native_calibration_shape_resolution"]["mode"] == "extrapolation"


def test_request_marker_shape_mismatch_does_not_reuse_exact_marker():
    profile = NativeCalibrationProfile(
        request_marker_ns={"first_token": 23.0},
        request_marker_policy="additive_once_per_request",
        request_marker_evidence={"first_token": {"status": "calibrated"}},
        request_shape={"prompt_tokens": 2, "output_tokens": 8, "prompt_fingerprint": "p2"},
    )
    assert request_marker_calibration_ns(profile, "first_token", prompt_tokens=3,
                                         output_tokens=8, prompt_fingerprint="p3") is None


def test_replay_evidence_contract_requires_token_boundaries_and_identity(tmp_path):
    from tools.replay_simulator_from_native import _validate_native_evidence
    trace = tmp_path / "native.log"; trace.write_text("kernel event\n", encoding="utf-8")
    perf = {"prompt": [{"ms": 1.0, "tokens": 2}], "decode": [{"ms": 2.0, "tokens": 2}], "total": [], "graphs_reused": []}
    import hashlib, json
    perf_sha = hashlib.sha256(json.dumps(perf, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    from heterollm_sim.serde import stable_hash
    prompt_fp = stable_hash("Hi")
    payload = {"configuration": {"ctx": 8, "parallel": 1, "batch": 4, "ubatch": 4, "threads": 1, "gpu_layers": -1},
               "request": {"prompt": "Hi", "prompt_fingerprint": prompt_fp, "output_mode": "fixed", "ignore_eos": True},
               "output_policy": {"mode": "fixed", "ignore_eos": True},
               "token_counts": {"output": 2},
               "identity": {"model_path": "model.gguf", "gguf_sha256": "m", "runtime_fingerprint": "r", "hardware_fingerprint": "h", "prompt_fingerprint": prompt_fp, "configuration": {"ctx": 8, "parallel": 1, "batch": 4, "ubatch": 4, "threads": 1, "gpu_layers": -1}},
               "native": {"perf_log": perf, "requests": [{"request_boundary": {"status": "measured", "token_chunk_times_ms": [1.0, 2.0], "request_to_first_token_ms": 1.0, "request_to_end_ms": 2.0}}]},
               "log": str(trace)}
    errors, summary = _validate_native_evidence(payload, source_path=tmp_path / "payload.json")
    assert errors == []
    assert summary["status"] == "complete"
    assert summary["evidence_level"] == "legacy_development"
    payload["native"]["requests"][0]["request_boundary"].pop("token_chunk_times_ms")
    errors, _ = _validate_native_evidence(payload, source_path=tmp_path / "payload.json")
    assert any("token timestamps missing" in item for item in errors)


def test_replay_evidence_contract_checks_supplemental_raw_trace_sha(tmp_path):
    from tools.replay_simulator_from_native import _validate_native_evidence
    trace = tmp_path / "native.log"; trace.write_text("server event\n", encoding="utf-8")
    nsys = tmp_path / "capture.nsys-rep"; nsys.write_bytes(b"nsys raw event")
    perf = {"prompt": [{"ms": 1.0, "tokens": 2}], "decode": [{"ms": 2.0, "tokens": 2}], "total": [], "graphs_reused": []}
    import hashlib, json
    perf_sha = hashlib.sha256(json.dumps(perf, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    ident = {"model_path": "model.gguf", "gguf_sha256": "m", "runtime_fingerprint": "r",
             "hardware_fingerprint": "h", "prompt_fingerprint": "p", "configuration":
             {"ctx": 8, "parallel": 1, "batch": 4, "ubatch": 4, "threads": 1, "gpu_layers": -1}}
    payload = {"configuration": ident["configuration"], "request": {"prompt": "Hi", "prompt_fingerprint": "p", "output_mode": "fixed", "ignore_eos": True},
               "output_policy": {"mode": "fixed", "ignore_eos": True}, "token_counts": {"output": 2}, "identity": ident,
               "native": {"perf_log": perf, "requests": [{"request_boundary": {"status": "measured", "token_chunk_times_ms": [1.0, 2.0], "request_to_first_token_ms": 1.0, "request_to_end_ms": 2.0}}]},
               "evidence": {"raw_trace_events": {"path": str(trace), "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                            "supplemental_artifacts": [{"path": str(nsys), "sha256": hashlib.sha256(nsys.read_bytes()).hexdigest()}]},
                            "extractor_output": {"sha256": perf_sha}}}
    errors, _ = _validate_native_evidence(payload, source_path=tmp_path / "payload.json")
    assert not any("supplemental raw trace" in item for item in errors)
    nsys.write_bytes(b"changed")
    errors, _ = _validate_native_evidence(payload, source_path=tmp_path / "payload.json")
    assert any("supplemental raw trace[0] sha256 mismatch" in item for item in errors)


def test_replay_evidence_contract_accepts_complete_manifest(tmp_path):
    from tools.native_llama_compare import native_extractor_identity
    from tools.replay_simulator_from_native import (
        _timing_contract, _validate_native_evidence, _native_measurements_digest,
    )
    from heterollm_sim.serde import stable_hash
    import hashlib, json, sys

    log = tmp_path / "native.log"; log.write_text("server event\n", encoding="utf-8")
    binary = Path(sys.executable)
    cfg = {"ctx": 8, "parallel": 1, "batch": 4, "ubatch": 4, "threads": 1, "threads_batch": 1,
           "gpu_layers": -1, "flash_attn": False, "mmap": True, "mlock": False, "offload_kqv": True,
           "op_offload": True, "split_mode": "layer", "main_gpu": 0, "cpu_range": None,
           "cpu_range_batch": None, "numa": None, "kv_type_k": "f16", "kv_type_v": "f16",
           "kv_unified": True, "continuous_batching": True, "coherent_dma_mode": "pipelined",
           "mtp": False, "temperature": 0.0, "top_k": 1, "seed": 42, "stop": [],
           "warmup_predict": 2, "request_timing": "stream"}
    identity = {"model_path": str((tmp_path / "model.gguf").resolve()), "gguf_sha256": "m",
                "runtime_fingerprint": "r", "hardware_fingerprint": "h",
                "prompt_fingerprint": stable_hash("Hi"),
                "configuration": {k: cfg[k] for k in ("ctx", "parallel", "batch", "ubatch", "threads", "gpu_layers")}}
    payload = {"command": [str(binary), "x"], "model": identity["model_path"], "configuration": cfg,
               "request": {"prompt": "Hi", "requested_output_tokens": 2, "output_mode": "fixed",
                           "ignore_eos": True, "warmup_output_tokens": 2,
                           "prompt_fingerprint": stable_hash("Hi")},
               "output_policy": {"mode": "fixed", "ignore_eos": True},
               "token_counts": {"prompt": 1, "output": 2, "requested_output": 2}, "runtime_config": {},
               "identity": identity, "hardware": {"cpu": "x", "gpu": "y", "llama_cpp": "0.3.0-dev build1 commit 0f3a71b"},
               "native": {"perf_log": {"prompt": [{"ms": 1, "tokens": 1}], "decode": [{"ms": 2, "tokens": 2}], "total": [], "graphs_reused": []},
                          "timings": {"prompt_ms": 1.0, "predicted_ms": 2.0},
                          "requests": [{"output_tokens": 2, "request_boundary": {"status": "measured", "token_chunk_times_ms": [1, 2], "request_to_first_token_ms": 1, "request_to_end_ms": 2}}]}}
    contract = _timing_contract()
    payload["evidence"] = {"schema": "native-evidence/v1",
        "native_binary": {"path": str(binary), "status": "captured", "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()},
        "extractor": native_extractor_identity(),
        "timing_contract": {**contract, "sha256": hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
        "engine_timing": {"status": "counter_proven", "source": "test.server_slot_stats",
                           "fields": ["engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms"]},
        "raw_trace_events": {"path": str(log), "status": "captured", "sha256": hashlib.sha256(log.read_bytes()).hexdigest()},
        "extractor_output": {"status": "captured", "field": "native.perf_log", "sha256": hashlib.sha256(json.dumps(payload["native"]["perf_log"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
        "token_timestamps": {"status": "captured", "field": "native.requests[].request_boundary.token_chunk_times_ms"},
        "request_boundaries": {"status": "captured", "field": "native.requests[].request_boundary"}}
    payload["evidence"]["native_contract_sha256"] = stable_hash({k: payload.get(k) for k in ("command", "configuration", "request", "output_policy", "token_counts", "runtime_config", "identity", "hardware")})
    payload["evidence"]["native_measurements_sha256"] = _native_measurements_digest(payload["native"])
    errors, summary = _validate_native_evidence(payload, source_path=tmp_path / "payload.json")
    assert errors == []
    assert summary["evidence_level"] == "complete"


def test_engine_replay_rejects_native_measurement_mutation():
    """A valid v3 payload cannot be turned into a new answer by editing counters."""
    import copy, json
    from tools.replay_simulator_from_native import _validate_native_evidence
    payload_path = (Path(__file__).resolve().parents[1] /
                    "artifacts/development/engine_contract_probe_engine_v3.json")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    evidence = payload["evidence"]
    if not evidence.get("native_measurements_sha256"):
        pytest.skip("development payload predates native measurement digest")
    mutated = copy.deepcopy(payload)
    mutated["native"]["aggregate"]["engine_e2e_ms"]["p50_ms"] += 1.0
    errors, _ = _validate_native_evidence(mutated, source_path=payload_path)
    assert "native measurements sha256 mismatch" in errors


def test_engine_v3_replay_requires_proven_engine_section():
    import json
    from tools.replay_simulator_from_native import _validate_native_evidence
    payload_path = (Path(__file__).resolve().parents[1] /
                    "artifacts/development/engine_contract_probe_engine_v3.json")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["evidence"].pop("engine_timing", None)
    errors, _ = _validate_native_evidence(payload, source_path=payload_path)
    assert "evidence.engine_timing not proven" in errors


def test_native_measurements_digest_excludes_outer_evidence():
    """Adding an evidence binding does not create a self-referential digest."""
    native = {"timings": {"prompt_ms": 1.0}, "aggregate": {"engine_e2e_ms": {"p50_ms": 1.0}}}
    digest = _native_measurements_digest(native)
    native["evidence"] = {"native_measurements_sha256": digest}
    assert _native_measurements_digest(native) == digest


def test_simulator_replay_artifacts_never_report_native_execution():
    """Replay row counts and native process counts are separate quantities."""
    import json
    root = Path(__file__).resolve().parents[1]
    for path in (root / "artifacts").rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("schema") == "simulator-replay-from-native/v1":
            assert payload.get("native_execution_count") == 0, path
