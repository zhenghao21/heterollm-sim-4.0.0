from tools.native_llama_profile import (
    _formal_request_records,
    _profile_coverage,
    _profiling_perturbation_declaration,
    _validity_status,
)


def _response(slot=None, predicted=2):
    result = {"timings": {"prompt_n": 3, "predicted_n": predicted, "prompt_ms": 1.0, "predicted_ms": 2.0}}
    if slot is not None:
        result["slot_id"] = slot
    return result


def _boundary():
    return {"status": "measured", "request_to_first_token_ms": 1.0, "request_to_end_ms": 2.0}


def test_profile_formal_records_preserve_request_slot_and_invocation_mapping():
    responses = [_response(4), _response(7)]
    responses[0]["invocation_ids"] = ["prefill-0", "decode-0"]
    responses[1]["invocation_ids"] = ["prefill-1", "decode-1"]
    records = _formal_request_records(responses, [_boundary(), _boundary()], 2)
    assert [item["request_id"] for item in records] == ["request-0000", "request-0001"]
    assert [item["slot_id"] for item in records] == [4, 7]
    assert all(item["slot_mapping_status"] == "observed" for item in records)
    assert all(item["invocation_mapping_status"] == "observed" for item in records)


def test_profile_coverage_fails_closed_when_kernel_owner_mapping_is_missing():
    responses = [_response(1), _response(2)]
    responses[0]["invocation_ids"] = ["p0", "d0"]
    responses[1]["invocation_ids"] = ["p1", "d1"]
    boundaries = [_boundary(), _boundary()]
    records = _formal_request_records(responses, boundaries, 2)
    stats = {
        "kernel": {"rows": [{"Name": "mul_mat_q", "Instances": "2"}]},
        "api": {"rows": [{"Name": "cudaLaunchKernel"}]},
        "memcpy": {"rows": []},
    }
    coverage = _profile_coverage(
        responses=responses,
        boundaries=boundaries,
        requested_parallel=2,
        requested_output=2,
        formal_records=records,
        stats=stats,
    )
    assert coverage["formal_request_count"] == 2
    assert coverage["request_count_status"] == "complete"
    assert coverage["phase_coverage"]["prompt_eval"]["status"] == "incomplete"
    assert coverage["phase_coverage"]["decode"]["status"] == "incomplete"
    assert coverage["request_mapping_status"] == "complete"
    assert coverage["owner_unverified_row_count"] == 1
    assert coverage["evidence_status"] == "incomplete"
    assert _validity_status(coverage=coverage, stats=stats, capture_mode="diagnostic") == "diagnostic_evidence_incomplete"


def test_profile_validity_does_not_accept_kernel_rows_without_request_evidence():
    stats = {"kernel": {"rows": [{"Name": "known"}]}}
    coverage = {"evidence_status": "incomplete"}
    assert _validity_status(coverage=coverage, stats=stats, capture_mode="diagnostic") == "diagnostic_evidence_incomplete"
    assert _validity_status(coverage=coverage, stats=stats, capture_mode="benchmark") == "benchmark_evidence_ineligible"
    assert _validity_status(coverage={"evidence_status": "complete"}, stats={"kernel": {"rows": []}}, capture_mode="diagnostic") == "evidence_missing"


def test_profile_declares_nsight_observer_effect():
    declaration = _profiling_perturbation_declaration("diagnostic")
    assert declaration["benchmark_eligible"] is False
    assert "nsys" in declaration["instrumentation"]
    assert declaration["expected_overhead"] == "nonzero_and_unmeasured"
