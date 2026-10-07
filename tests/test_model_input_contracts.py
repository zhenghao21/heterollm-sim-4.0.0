import http.client
import json
from dataclasses import replace
import threading

from heterollm_sim.config import HostOutputContract, scenario_from_dict
from heterollm_sim.llama_scenario import prepare_llama_scenario
from heterollm_sim.planner import validate_scenario
from heterollm_sim.reference import build_llama_default_scenario, build_reference_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.web import build_server, scenario_to_payload


def test_normalized_runtime_payload_is_authoring_round_trippable():
    source = replace(
        build_llama_default_scenario(),
        llama_cpp_config=LlamaCppRuntimeConfig(),
    )
    prepared = prepare_llama_scenario(source)
    payload = scenario_to_payload(prepared)

    # Runtime placement is derived state and must not leak into V4 authoring.
    assert payload["placement"]["op_to_component"] == {}
    assert payload["placement"]["tensor_to_component"] == {}
    assert payload["placement"]["tensor_bytes"] == {}

    reloaded = scenario_from_dict(payload)
    reparsed = prepare_llama_scenario(reloaded)
    assert reparsed.placement.op_to_component
    assert reparsed.placement.tensor_bytes
    assert validate_scenario(reparsed).is_valid


def test_host_output_vocabulary_mismatch_is_rejected_before_costing():
    source = build_reference_scenario()
    scenario = replace(
        source,
        host_output_contract=HostOutputContract(
            target_component_id="cpu0",
            vocabulary_size=1,
            logits_dtype="int8",
            logits_element_bytes=1,
        ),
    )
    report = validate_scenario(scenario)
    assert not report.is_valid
    assert any("vocabulary_size" in message for message in report.errors_en)


def test_missing_physical_memory_config_is_rejected_at_web_boundaries():
    payload = scenario_to_payload(build_llama_default_scenario())
    components = payload["hardware_input"]["hardware"]["components"]
    memory = next(item for item in components if item["kind"] == "host_memory")
    memory["metadata"].pop("physical_memory_config")

    server = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]

        def post(path, body):
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request(
                "POST", path, body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            result = json.loads(response.read().decode("utf-8"))
            connection.close()
            return response.status, result

        normalize_status, normalized = post("/api/normalize", payload)
        assert normalize_status == 422
        assert "physical_memory_config" in normalized["error"]["message_en"]

        validate_status, validation = post("/api/validate", payload)
        assert validate_status == 200
        assert validation["valid"] is False
        assert "physical_memory_config" in validation["errors"]["scenario"][0]["message_en"]

        run_status, run_result = post("/api/run-jobs", {"scenario": payload})
        assert run_status == 422
        assert "physical_memory_config" in run_result["error"]["message_en"]
        assert server.run_job_manager.list(status="running") == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
