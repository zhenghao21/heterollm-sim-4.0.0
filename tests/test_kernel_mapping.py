from heterollm_sim.kernel_mapping import classify_kernel, map_kernel_profile, STAGES

def test_stage_markers_and_unknown_are_conservative():
    assert classify_kernel("rms_norm_f32").stage == "normalization"
    assert classify_kernel("q_proj_matmul").stage == "attention_qkv"
    assert classify_kernel("some_generic_kernel").stage == "unknown"
    assert classify_kernel("mul_mat", {"graph_stage": "ffn"}).confidence == 1.0


def test_semantic_operator_owner_refines_generic_mmq_stage():
    assert classify_kernel("mul_mat_vec_q", {"stage": "unknown", "semantic_operator_id": "MUL_MAT:Qcur-3"}).stage == "attention_qkv"
    assert classify_kernel("mul_mat_q", {"stage": "unknown", "semantic_operator_id": "MUL_MAT:ffn_gate-3"}).stage == "ffn"
    assert classify_kernel("mul_mat_vec_q", {"stage": "unknown", "semantic_operator_id": "MUL_MAT:result_output"}).stage == "lm_head"
    assert classify_kernel("mul_mat_q", {"stage": "unknown", "semantic_operator_id": "MUL_MAT:node_31"}).stage == "unknown"


def test_qwen35_auxiliary_and_full_attention_owners_stay_separate():
    assert classify_kernel("generic", {"stage": "unknown", "semantic_operator_id": "SSM_CONV:conv_output_raw-0"}).stage == "linear_attention_aux"
    assert classify_kernel("generic", {"stage": "unknown", "semantic_operator_id": "GATED_DELTA_NET:node_46"}).stage == "linear_attention_aux"
    assert classify_kernel("generic", {"stage": "unknown", "semantic_operator_id": "SCALE:q_conv_predelta-0"}).stage == "linear_attention_aux"
    assert classify_kernel("generic", {"stage": "unknown", "semantic_operator_id": "CONCAT:conv_input-0"}).stage == "linear_attention_aux"
    assert classify_kernel("generic", {"stage": "unknown", "semantic_operator_id": "CPY:conv_state_update-0"}).stage == "linear_attention_aux"
    assert classify_kernel("generic", {"stage": "unknown", "semantic_operator_id": "CONT:attn_pregate-3"}).stage == "attention_output"

def test_aggregate_preserves_unknown_and_shares():
    result = map_kernel_profile([{"Name": "q_proj", "Total Time (ns)": "300", "Instances": "3", "Avg (ns)": "100"}, {"Name": "unknown", "Total Time (ns)": "700", "Instances": "1"}])
    assert set(result["aggregate"]) == set(STAGES)
    assert result["aggregate"]["attention_qkv"]["total_ns"] == 300
    assert sum(v["share"] for v in result["aggregate"].values()) == 1.0
