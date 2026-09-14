from tools.native_error_matrix import CASES, _derive_gates, _error_summary

def test_matrix_has_length_output_batch_and_placement_coverage():
    ids={c['id'] for c in CASES}
    assert {'short_output1','long_output8','medium_output32','batch32_ub16','cpu_only','gpu_tail12'} <= ids
    assert all(c['ubatch'] <= c['batch'] for c in CASES)


def test_matrix_gates_keep_token_and_timing_claims_separate():
    payload = {
        "parity": {"geometry": {"ok": True}, "tokens": {
            "ok": True, "native_prompt": 2, "simulator_prompt": 2,
            "native_output": 8, "simulator_output": 8,
        }},
        "configuration": {"ctx": 512, "parallel": 1, "batch": 64,
                           "ubatch": 64, "threads": 16, "gpu_layers": -1},
        "runtime_config": {"context": 512, "parallel": 1, "batch": 64,
                            "ubatch": 64, "threads": 16, "gpu_layers": -1},
    }
    gates = _derive_gates(payload)
    assert gates["geometry_gate"] == "pass"
    assert gates["token_count_gate"] == "structural_only"
    assert gates["token_count_gate_detail"]["independent"] is False
    assert gates["timing_gate"] == "diagnostic_only"
    assert gates["boundary_mismatch"]["ttft"] is True
    assert gates["overall_status"] == "valid_for_structural_analysis"


def test_error_summary_excludes_undefined_tpot():
    rows = [
        {"relative_error_pct": {"ttft_ms": 10.0, "tpot_ms": None, "e2e_ms": 20.0}},
        {"relative_error_pct": {"ttft_ms": 30.0, "tpot_ms": 40.0, "e2e_ms": 50.0}},
    ]
    summary = _error_summary(rows)
    assert summary["tpot_ms"]["n"] == 1
    assert summary["tpot_ms"]["median_pct"] == 40.0
