import http.client
import json
from pathlib import Path
import threading

from heterollm_sim.config import model_from_dict
from heterollm_sim.model_artifacts import (
    MODEL_ARTIFACT_SCHEMA,
    list_model_artifacts,
    load_model_artifact,
    save_model_artifact,
)
from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.serde import to_primitive
from heterollm_sim.web import build_server, scenario_to_payload


def test_model_artifact_round_trip_and_artifact_id_resolution(tmp_path, monkeypatch):
    model = build_llama_default_scenario().model
    document = save_model_artifact(
        model,
        artifact_dir=tmp_path,
        artifact_id="round-trip-model",
        provenance={"source_type": "test"},
    )

    assert document["schema_version"] == MODEL_ARTIFACT_SCHEMA
    loaded = load_model_artifact("round-trip-model", artifact_dir=tmp_path)
    assert loaded["model"]["name"] == model.name
    assert len(list_model_artifacts(artifact_dir=tmp_path)) == 1

    monkeypatch.setenv("HETEROLLM_SIM_MODEL_ARTIFACT_DIR", str(tmp_path))
    resolved = model_from_dict({
        "schema_version": "4.0.0",
        "artifact_id": "round-trip-model",
    })
    assert resolved.name == model.name
    assert resolved.metadata["artifact_id"] == "round-trip-model"


def test_model_file_api_saves_and_reads_graph(tmp_path):
    server = build_server(
        host="127.0.0.1",
        port=0,
        model_artifact_dir=tmp_path,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        connection = http.client.HTTPConnection(host, port, timeout=5)
        body = json.dumps({"model": to_primitive(build_llama_default_scenario().model)})
        connection.request(
            "POST",
            "/api/model-files",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        assert response.status == 201
        artifact_id = payload["artifact_id"]
        connection.close()

        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request("GET", "/api/model-files/{}".format(artifact_id))
        response = connection.getresponse()
        loaded = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert loaded["artifact_id"] == artifact_id
        assert loaded["model"]["graph"]["executable"] is True
        connection.close()

        scenario_payload = scenario_to_payload(build_llama_default_scenario())
        scenario_payload["model"] = {
            "schema_version": "4.0.0",
            "artifact_id": artifact_id,
        }
        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request(
            "POST",
            "/api/normalize",
            body=json.dumps(scenario_payload),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        normalized = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert normalized["model"]["graph"]["executable"] is True
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
