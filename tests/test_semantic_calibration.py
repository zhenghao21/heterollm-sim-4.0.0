import json

from tools.build_semantic_calibration import build, classify_operator, _operator_id, _phase_scopes, _event_phase
from heterollm_sim.calibration import load_native_calibration
from heterollm_sim.kernel_mapping import classify_kernel


def _trace(path, events):
    path.write_text(json.dumps({"schema": "native-nsys-trace/v1", "events": events}), encoding="utf-8")


def test_explicit_operator_rates_and_categories(tmp_path):
    assert classify_operator("Qcur") == "attention_qkv"
    assert classify_operator("ffn_gate") == "ffn"
    assert classify_operator("cache_k_l0/SET_ROWS") == "kv"
    assert classify_operator("result_output/GET_ROWS") == "lm_head"
    assert classify_operator("MUL_MAT:kqv_wo-0") == "attention_output"
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    labels = ["Qcur", "ffn_gate", "cache_k_l0/SET_ROWS", "result_output/GET_ROWS"]
    _trace(train, [{"kind": "kernel", "duration_ns": 10, "semantic_status": "matched", "semantic_operator": label} for label in labels])
    _trace(holdout, [{"kind": "kernel", "duration_ns": 10, "semantic_status": "matched", "semantic_operator": label} for label in labels])
    result = build(train, holdout)
    assert result["schema"] == "native-semantic-calibration/v1"
    assert all(result["stages"][stage]["status"] == "calibrated" for stage in ("attention_qkv", "ffn", "kv", "lm_head"))
    assert all(result["stages"][stage]["holdout_relative_error_pct"] == 0.0 for stage in ("attention_qkv", "ffn", "kv", "lm_head"))
    assert classify_kernel("k_bin_bcast", {"semantic_operator_id": "ADD:ffn_inp-0"}).stage == "ffn"


def test_lifecycle_markers_do_not_create_ambiguous_phase_scope():
    payload = {"nvtx_events": [
        {"label": "prefill_begin|slot=0", "start_ns": 10, "end_ns": 100, "global_tid": 7},
        {"label": "phase:prefill", "start_ns": 20, "end_ns": 90, "global_tid": 7},
    ]}
    scopes = _phase_scopes(payload)
    assert len(scopes) == 1
    event = {"runtime_start_ns": 30, "runtime_end_ns": 40, "runtime_global_tid": 7}
    assert _event_phase(event, scopes) == "prefill"


def test_qwen35_linear_attention_aux_owners_are_separate(tmp_path):
    assert classify_operator("SSM_CONV:conv_output_raw-0") == "linear_attention_aux"
    assert classify_operator("GATED_DELTA_NET:node_46") == "linear_attention_aux"
    assert classify_operator("SCALE:q_conv_predelta-0") == "linear_attention_aux"
    assert classify_operator("CONCAT:conv_input-0") == "linear_attention_aux"
    assert classify_operator("CPY:conv_state_update-0") == "linear_attention_aux"
    assert classify_operator("UNARY:beta_sigmoid-0") == "linear_attention_aux"
    assert classify_operator("CONT:attn_pregate-3") == "attention_output"
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    event = {"kind": "kernel", "duration_ns": 10, "semantic_status": "matched",
             "semantic_operator": "SSM_CONV:conv_output_raw-0", "phase": "decode", "token_shape": "1024x1x1x1"}
    _trace(train, [event]); _trace(holdout, [event])
    result = build(train, holdout)
    aux = result["stages"]["linear_attention_aux"]
    assert aux["status"] == "calibrated"
    assert aux["holdout_relative_error_pct"] == 0.0


def test_v2_profile_with_aux_stage_loads(tmp_path):
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    event = {"kind": "kernel", "duration_ns": 10, "semantic_status": "matched",
             "semantic_operator": "SSM_CONV:conv_output_raw-0", "phase": "decode", "token_shape": "1x1"}
    _trace(train, [event]); _trace(holdout, [event])
    profile_path = tmp_path / "profile_v2.json"
    profile_path.write_text(json.dumps(build(train, holdout, schema="native-semantic-calibration/v2")), encoding="utf-8")
    profile = load_native_calibration(profile_path)
    assert profile is not None
    assert profile.kernel_stage_mapping["linear_attention_aux"]["status"] == "calibrated"


def test_unknown_and_uncovered_are_blocked(tmp_path):
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    _trace(train, [{"kind": "kernel", "duration_ns": 10, "semantic_status": "matched", "semantic_operator": "mystery"}])
    _trace(holdout, [{"kind": "kernel", "duration_ns": 10, "semantic_status": "unknown"}])
    result = build(train, holdout)
    assert result["coverage"]["status"] == "blocked"
    assert result["coverage"]["train_unknown_or_uncovered_count"] == 0
    assert result["coverage"]["train_explicit_excluded_count"] == 1
    assert result["stages"]["attention_qkv"]["status"] == "blocked_no_semantic_evidence"


def test_phase_and_token_shape_groups_are_calibrated_independently(tmp_path):
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    train_events = [
        {"kind": "kernel", "duration_ns": 10, "semantic_status": "matched", "semantic_operator": "Qcur", "phase": "prefill", "token_shape": "b1_t4"},
        {"kind": "kernel", "duration_ns": 20, "semantic_status": "matched", "semantic_operator": "Qcur", "phase": "prefill", "token_shape": "b1_t8"},
        {"kind": "kernel", "duration_ns": 30, "semantic_status": "matched", "semantic_operator": "Qcur", "phase": "decode", "token_shape": {"batch": 1, "tokens": 1}},
    ]
    holdout_events = [
        {"kind": "kernel", "duration_ns": 15, "semantic_status": "matched", "semantic_operator": "Qcur", "phase": "prefill", "token_shape": "b1_t4"},
        {"kind": "kernel", "duration_ns": 25, "semantic_status": "matched", "semantic_operator": "Qcur", "phase": "prefill", "token_shape": "b1_t8"},
        {"kind": "kernel", "duration_ns": 30, "semantic_status": "matched", "semantic_operator": "Qcur", "phase": "decode", "token_shape": {"batch": 1, "tokens": 1}},
    ]
    _trace(train, train_events)
    _trace(holdout, holdout_events)
    result = build(train, holdout)
    prefill = result["stages"]["attention_qkv"]["phases"]["prefill"]
    assert prefill["train_instances"] == 2
    assert prefill["holdout_instances"] == 2
    assert prefill["train_ns_per_instance"] == 15.0
    assert prefill["holdout_ns_per_instance"] == 20.0
    assert prefill["holdout_relative_error_pct"] == -25.0
    assert prefill["token_shapes"]["b1_t4"]["train_instances"] == 1
    assert prefill["token_shapes"]["b1_t4"]["holdout_relative_error_pct"] == (10.0 - 15.0) / 15.0 * 100.0
    assert result["stages"]["attention_qkv"]["phases"]["decode"]["train_instances"] == 1


def test_graph_replay_and_missing_shape_remain_blocked(tmp_path):
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    event = {"kind": "kernel", "duration_ns": 10, "semantic_status": "matched", "semantic_operator": "Qcur", "phase": "prefill", "graph_node_id": 7}
    _trace(train, [event])
    _trace(holdout, [{**event, "graph_node_id": None}])
    result = build(train, holdout)
    assert result["coverage"]["status"] == "blocked"
    assert result["coverage"]["train_unknown_or_uncovered_count"] == 1
    unknown_shape = result["stages"]["attention_qkv"]["phases"]["prefill"]["token_shapes"]
    assert unknown_shape["unknown"]["status"] == "blocked_no_semantic_evidence"


def test_calibration_uses_stable_operator_id_and_excludes_memcpy(tmp_path):
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    label = "MUL_MAT:Qcur-0|shape=896x6x1x1|type=f32"
    events = [
        {"kind": "kernel", "duration_ns": 10, "semantic_status": "matched",
         "semantic_operator": label, "semantic_operator_id": "MUL_MAT:Qcur-0",
         "phase": "prefill", "token_shape": "896x6x1x1"},
        {"kind": "memcpy", "duration_ns": 100, "semantic_status": "matched",
         "semantic_operator": label, "semantic_operator_id": "MUL_MAT:Qcur-0"},
    ]
    _trace(train, events)
    _trace(holdout, [{**events[0], "duration_ns": 20}, events[1]])
    result = build(train, holdout)
    stage = result["stages"]["attention_qkv"]
    assert stage["train_total_ns"] == 10
    assert stage["holdout_total_ns"] == 20
    assert list(stage["operators"]) == ["MUL_MAT:Qcur-0"]
    assert result["coverage"]["train_memcpy_excluded_count"] == 1
    assert result["coverage"]["train_memcpy_excluded_time_ns"] == 100


def test_explicit_unsupported_stage_is_excluded_without_blocking(tmp_path):
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    labels = ["Qcur", "ffn_gate", "cache_k_l0/SET_ROWS", "result_output/GET_ROWS", "MYSTERY_STAGE"]
    events = [{"kind": "kernel", "duration_ns": 10, "semantic_status": "matched", "semantic_operator": label,
               "phase": "decode", "token_shape": "1x1"} for label in labels]
    _trace(train, events)
    _trace(holdout, events)
    result = build(train, holdout)
    assert result["coverage"]["status"] == "covered"
    assert result["coverage"]["train_explicit_excluded_count"] == 1
    assert all(result["stages"][stage]["status"] == "calibrated" for stage in ("attention_qkv", "ffn", "kv", "lm_head"))


def test_identity_mismatch_blocks_coverage(tmp_path):
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    event = {"kind": "kernel", "duration_ns": 10, "semantic_status": "matched",
             "semantic_operator": "Qcur", "phase": "decode", "token_shape": "1x1"}
    train.write_text(json.dumps({"schema": "native-nsys-trace/v1", "trace_identity": {"model_sha256": "a"}, "events": [event]}), encoding="utf-8")
    holdout.write_text(json.dumps({"schema": "native-nsys-trace/v1", "trace_identity": {"model_sha256": "b"}, "events": [event]}), encoding="utf-8")
    result = build(train, holdout)
    assert result["coverage"]["identity_match"] is False
    assert result["coverage"]["identity_mismatch_fields"] == ["model_sha256"]
    assert result["coverage"]["status"] == "blocked"


def test_operator_wall_uses_longest_nested_scope(tmp_path):
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    def payload(parent_end, child_end):
        return {"schema": "native-nsys-trace/v1", "events": [
            {"kind": "kernel", "duration_ns": 5, "semantic_status": "matched", "semantic_operator": "Qcur", "semantic_operator_id": "Qcur", "phase": "decode", "token_shape": "1x1"}
        ], "nvtx_events": [
            {"operator_name": "Qcur", "operator_id": "Qcur", "label": "operator:Qcur|shape=1x1", "start_ns": 0, "end_ns": parent_end, "global_tid": 1},
            {"operator_name": "Qcur", "operator_id": "Qcur", "label": "operator:Qcur|shape=1x1", "start_ns": 1, "end_ns": child_end, "global_tid": 1},
            {"phase_name": "prefill", "label": "phase:prefill", "start_ns": 0, "end_ns": 100, "global_tid": 1},
        ]}
    train.write_text(json.dumps(payload(20, 10)), encoding="utf-8")
    holdout.write_text(json.dumps(payload(30, 15)), encoding="utf-8")
    result = build(train, holdout)
    stage = result["stages"]["attention_qkv"]
    assert result["calibration_basis"] == "operator_wall"
    assert stage["wall_train_instances"] == 1
    assert stage["wall_ns_per_instance"] == 20
    assert stage["phases"]["prefill"]["wall_token_shapes"]["1x1"]["wall_ns_per_instance"] == 20


def test_operator_wall_label_parser_is_stable():
    assert _operator_id("operator:MUL_MAT:Qcur-0|shape=896x6x1x1|type=f32") == "MUL_MAT:Qcur-0"
    assert _operator_id("[MUL_MAT:Qcur-0]|shape=896x1x1x1") == "MUL_MAT:Qcur-0"
    assert _operator_id("") is None
