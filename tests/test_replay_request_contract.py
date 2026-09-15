"""Synthetic request-shape and frozen prediction scoring checks; no native runs."""
import copy
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import replay_simulator_from_native as replay_module


def _case(parallel, output=5):
    native, simulated = [], []
    for index in range(parallel):
        ttft, tpot = 10.0 + index, 2.0 + index / 2
        record = {"request_id": f"request-{index:04d}", "prompt_tokens": 8, "output_tokens": output,
                  "engine_timing_status": "counter_proven", "timing_contract_id": replay_module.ENGINE_CONTRACT_ID,
                  "engine_ttft_ms": ttft, "engine_tpot_ms": tpot if output > 1 else None,
                  "engine_e2e_ms": ttft + tpot * (output - 1)}
        native.append(record)
        predicted = {**record, "visible_output_tokens": output, "engine_timing_status": "measured"}
        for field in ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms"):
            if predicted[field] is not None:
                predicted[field] *= 1.1
        simulated.append(predicted)
    payload = {"model": "synthetic.gguf", "configuration": {"parallel": parallel},
               "token_counts": {"prompt": 8, "output": output},
               "request": {"prompt": "synthetic prompt", "requested_output_tokens": output, "output_mode": "fixed"},
               "native": {"requests": native, "aggregate": replay_module._aggregate_request_records(native)},
               "parallel_support": {"requested": parallel, "native_requests": parallel}}
    prediction = {"schema": "simulator-replay-prediction/v1", "calibration_mode": "analytical", "simulator_requests": simulated}
    return payload, prediction


@pytest.mark.parametrize("parallel", [2, 4])
def test_recompute_parallel_request_sets_and_ignore_cached_metrics(parallel):
    payload, prediction = _case(parallel)
    payload["native"]["aggregate"]["engine_ttft_ms"]["p50_ms"] = 99999.0
    prediction["metrics"] = {"ttft_ms": {"simulator_ms": -1}}
    prediction["simulator_requests"].reverse()
    before = copy.deepcopy((payload, prediction))
    result = replay_module.score_saved_prediction(payload, prediction)
    assert result["status"] == "valid", result
    assert result["engine_evaluation"]["status"] == "measured"
    assert result["native"]["engine_ttft_ms"] == statistics.median(10.0 + index for index in range(parallel))
    assert all(value == pytest.approx(10.0) for value in result["relative_error_pct"].values())
    assert result["simulator_aggregate"]["record_scope"] == "request_set"
    assert result["simulator_aggregate"]["request_ids"] == [record["request_id"] for record in prediction["simulator_requests"]]
    assert result["simulator_aggregate"]["request_count"] == parallel
    assert (payload, prediction) == before


@pytest.mark.parametrize("side", ["native", "simulator"])
@pytest.mark.parametrize("mutation", ["missing_request", "duplicate_id", "wrong_id", "missing_id", "wrong_output", "bool_output", "wrong_prompt"])
def test_request_shapes_fail_closed(side, mutation):
    payload, prediction = _case(4)
    records = payload["native"]["requests"] if side == "native" else prediction["simulator_requests"]
    if mutation == "missing_request":
        records.pop()
    elif mutation == "duplicate_id":
        records[-1]["request_id"] = records[0]["request_id"]
    elif mutation == "wrong_id":
        records[-1]["request_id"] = "other-request"
    elif mutation == "missing_id":
        records[-1].pop("request_id")
    elif mutation == "wrong_output":
        records[-1]["output_tokens"] -= 1
    elif mutation == "bool_output":
        records[-1]["output_tokens"] = True
    else:
        records[-1]["prompt_tokens"] += 1
    result = replay_module.score_saved_prediction(payload, prediction)
    assert result["status"] == "invalid"
    assert result["request_errors"]
    assert result["engine_evaluation"]["status"] == "evidence_insufficient"
    assert all(value is None for value in result["relative_error_pct"].values())


@pytest.mark.parametrize("mutation", ["native_count", "native_ids", "saved_count", "support_count", "bool_parallel"])
def test_declared_request_counters_cannot_hide_real_request_set(mutation):
    payload, prediction = _case(2)
    if mutation == "native_count":
        payload["native"]["aggregate"]["request_count"] = 1
    elif mutation == "native_ids":
        payload["native"]["aggregate"]["request_ids"] = ["unrelated"]
    elif mutation == "saved_count":
        prediction["simulator_request_count"] = 1
    elif mutation == "support_count":
        payload["parallel_support"]["native_requests"] = 1
    else:
        payload["configuration"]["parallel"] = True
    assert replay_module.score_saved_prediction(payload, prediction)["status"] == "invalid"


@pytest.mark.parametrize("side", ["native", "simulator"])
@pytest.mark.parametrize("mutation", ["negative", "nan", "infinity", "end_before_first", "tpot_disagrees", "missing_tpot", "bad_status"])
def test_invalid_engine_timing_is_not_scored(side, mutation):
    payload, prediction = _case(2)
    record = (payload["native"]["requests"] if side == "native" else prediction["simulator_requests"])[0]
    if mutation in {"negative", "nan", "infinity"}:
        record["engine_ttft_ms"] = {"negative": -1.0, "nan": float("nan"), "infinity": float("inf")}[mutation]
    elif mutation == "end_before_first":
        record["engine_e2e_ms"] = record["engine_ttft_ms"] - 1
    elif mutation == "tpot_disagrees":
        record["engine_tpot_ms"] *= 2
    elif mutation == "missing_tpot":
        record.pop("engine_tpot_ms")
    else:
        record["engine_timing_status"] = {}
    result = replay_module.score_saved_prediction(payload, prediction)
    assert result["status"] == "invalid"
    assert result["request_errors"]


def test_single_token_output_preserves_tpot_not_applicable():
    payload, prediction = _case(2, output=1)
    result = replay_module.score_saved_prediction(payload, prediction)
    assert result["status"] == "valid", result
    assert result["engine_evaluation"]["metrics"]["tpot_ms"]["status"] == "not_applicable"
    assert result["simulator"]["engine_tpot_ms"] is None


@pytest.mark.parametrize("parallel", [2, 4])
def test_replay_saves_real_request_ids_and_frozen_prediction_identity(tmp_path, monkeypatch, parallel):
    payload, expected = _case(parallel)
    requests = {}
    for record in expected["simulator_requests"]:
        first = (1.0 + record["engine_ttft_ms"]) * 1e6
        last = (1.0 + record["engine_e2e_ms"]) * 1e6
        # The client finish is later than the last engine token; engine TPOT
        # must use the engine token boundary instead of this client counter.
        requests[record["request_id"]] = SimpleNamespace(
            request_id=record["request_id"], arrival_ns=0.0, start_ns=1e6,
            first_token_ns=first, last_token_ns=last, finish_ns=last + 5e6,
            tpot_ns=(last + 5e6 - first) / 4, visible_output_tokens=5)
    simulation = SimpleNamespace(metrics=SimpleNamespace(request_metrics=requests),
                                 serving=SimpleNamespace(request_metrics={}, events=[]))
    scenario = SimpleNamespace(workload=SimpleNamespace(scheduler=SimpleNamespace(prefill_chunk_tokens=8)))
    monkeypatch.setattr(replay_module, "_validate_native_evidence", lambda *a, **kw: ([], {"evidence_level": "complete"}))
    monkeypatch.setattr(replay_module, "read_gguf_metadata", lambda path: {})
    monkeypatch.setattr(replay_module, "build_model_from_gguf", lambda metadata: object())
    monkeypatch.setattr(replay_module, "build_matching_scenario", lambda *a, **kw: scenario)
    monkeypatch.setattr(replay_module, "run_scenario", lambda *a, **kw: simulation)
    source, output = tmp_path / "source.json", tmp_path / "prediction.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    identity = {"freeze_sha256": "f" * 64, "cell_id": f"synthetic__p{parallel}__r1"}
    result = replay_module.replay(payload, "synthetic", source, calibration_mode="analytical",
                                  prediction_output=output, prediction_identity=identity)
    assert result["status"] == "valid", result
    assert all(value == pytest.approx(10.0) for value in result["relative_error_pct"].values())
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["freeze_sha256"] == identity["freeze_sha256"]
    assert saved["cell_id"] == identity["cell_id"]
    assert saved["calibration_mode"] == "analytical"
    assert saved["source_payload_sha256"] == replay_module._sha256_file(source)
    assert saved["independent_blind_prediction"] is False
    assert [record["request_id"] for record in saved["simulator_requests"]] == list(requests)
    rescored = replay_module.score_saved_prediction(payload, saved)
    assert rescored["native"] == result["native"]
    assert rescored["simulator"] == result["simulator"]
    assert rescored["engine_evaluation"] == result["engine_evaluation"]
    with pytest.raises(FileExistsError):
        replay_module.replay(payload, "synthetic", source, calibration_mode="analytical",
                             prediction_output=output, prediction_identity=identity)


@pytest.mark.parametrize("identity", [
    {"freeze_sha256": "f" * 64, "cell_id": "cell", "schema": "override"},
    {"freeze_sha256": "f" * 64, "cell_id": "cell", "calibration_mode": "legacy"},
    {"freeze_sha256": "f" * 64, "cell_id": "cell", "source_payload_sha256": "a" * 64},
    {"freeze_sha256": "", "cell_id": "cell"},
    {"freeze_sha256": "f" * 64, "cell_id": ""},
])
def test_prediction_identity_cannot_override_reserved_fields(tmp_path, identity):
    output = tmp_path / "prediction.json"
    with pytest.raises(ValueError, match="prediction_identity"):
        replay_module.replay({}, "synthetic", calibration_mode="analytical", prediction_output=output,
                             prediction_identity=identity)
    assert not output.exists()
