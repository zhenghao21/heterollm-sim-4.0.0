"""Real HTTP requests enforce remote authentication separately from CORS."""

from http.client import HTTPConnection
import json
import threading

import pytest

from heterollm_sim import web


TOKEN = "test-backend-access-token"
ORIGIN = "https://zh.gripe"


@pytest.fixture
def api_server(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "health_payload", lambda version: {
        "ok": True, "version": version, "runtime": {"python_version": "test"},
    })
    monkeypatch.setattr(web, "validation_payload", lambda payload, **kwargs: {
        "valid": True, "request": payload,
    })
    servers = []

    def start(*, remote=True, trusted_origins=(ORIGIN, "https://www.zh.gripe")):
        index = str(len(servers))
        server = web.build_server(
            "127.0.0.1", 0,
            catalog_cache_dir=tmp_path / index / "models",
            component_preset_cache_dir=tmp_path / index / "components",
            model_artifact_dir=tmp_path / index / "artifacts",
            remote_access_token=TOKEN if remote else None,
            trusted_origins=trusted_origins,
        )
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def request(server, method="GET", path="/api/health", *, headers=None, body=None):
    client = HTTPConnection(*server.server_address[:2], timeout=5)
    try:
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse()
        raw = response.read()
        return response.status, dict(response.getheaders()), json.loads(raw) if raw else None
    finally:
        client.close()


@pytest.mark.parametrize("path", [
    "/api/health", "/api/reference", "/api/model-files", "/api/component-presets",
    "/api/run-jobs/private-job/trace", "/api/not-found", "/api", "/api/",
])
def test_all_remote_api_gets_require_authentication(api_server, path):
    status, headers, payload = request(api_server(), path=path)
    assert status == 401
    assert payload["error"]["code"] == "authentication_required"
    assert headers["WWW-Authenticate"] == "Bearer"
    assert "Access-Control-Allow-Origin" not in headers


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_remote_writes_require_authentication_before_routing_or_body_parse(api_server, method):
    status, _, payload = request(api_server(), method, "/api/component-presets/private", body="invalid-json")
    assert status == 401
    assert payload["error"]["code"] == "authentication_required"


def test_trusted_origin_does_not_replace_authentication_and_error_has_cors(api_server):
    server = api_server()
    for authorization in (None, "Bearer wrong", "Basic " + TOKEN, "Bearer " + TOKEN + " "):
        headers = {"Origin": ORIGIN}
        if authorization is not None:
            headers["Authorization"] = authorization
        status, returned_headers, payload = request(server, headers=headers)
        assert status == 401
        assert payload["error"]["code"] == "authentication_required"
        assert returned_headers["Access-Control-Allow-Origin"] == ORIGIN
        assert returned_headers["Vary"] == "Origin"
        assert TOKEN not in json.dumps(payload)


@pytest.mark.parametrize("origin", [ORIGIN, "https://www.zh.gripe", None])
def test_authorized_remote_health_keeps_existing_contract(api_server, origin):
    headers = {"Authorization": "Bearer " + TOKEN}
    if origin is not None:
        headers["Origin"] = origin
    status, returned_headers, payload = request(api_server(), headers=headers)
    assert status == 200
    assert payload == {"ok": True, "version": web.__version__, "runtime": {"python_version": "test"}}
    assert returned_headers.get("Access-Control-Allow-Origin") == origin
    assert "Access-Control-Allow-Credentials" not in returned_headers


def test_authorized_trusted_cross_site_write_is_allowed(api_server):
    status, headers, payload = request(api_server(), "POST", "/api/validate", headers={
        "Authorization": "Bearer " + TOKEN, "Origin": ORIGIN,
        "Sec-Fetch-Site": "cross-site", "Content-Type": "application/json",
    }, body='{"test": true}')
    assert status == 200
    assert payload == {"valid": True, "request": {"test": True}}
    assert headers["Access-Control-Allow-Origin"] == ORIGIN


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://zh.gripe", "https://zh.gripe.evil.example"])
def test_untrusted_origins_are_denied_even_with_valid_token_or_forwarded_headers(api_server, origin):
    status, headers, payload = request(api_server(), headers={
        "Origin": origin, "Authorization": "Bearer " + TOKEN,
        "X-Forwarded-Host": "zh.gripe", "X-Forwarded-Proto": "https",
    })
    assert status == 403
    assert payload["error"]["code"] == "cross_origin_forbidden"
    assert "Access-Control-Allow-Origin" not in headers
    assert origin not in json.dumps(payload)


def test_token_is_still_required_when_remote_allowlist_is_empty(api_server):
    server = api_server(trusted_origins=())
    assert request(server)[0] == 401
    assert request(server, headers={"Authorization": "Bearer " + TOKEN})[0] == 200
    assert request(server, headers={"Authorization": "Bearer " + TOKEN, "Origin": ORIGIN})[0] == 403


def test_trusted_preflight_needs_no_token_and_has_fixed_minimal_permissions(api_server):
    status, headers, payload = request(api_server(), "OPTIONS", "/api/run-jobs", headers={
        "Origin": ORIGIN, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization, content-type",
    })
    assert status == 204 and payload is None
    assert headers["Access-Control-Allow-Origin"] == ORIGIN
    assert set(headers["Access-Control-Allow-Methods"].split(", ")) == web.REMOTE_API_METHODS
    assert headers["Access-Control-Allow-Headers"] == "Authorization, Content-Type"
    assert "Access-Control-Allow-Credentials" not in headers


@pytest.mark.parametrize("origin,method,requested_headers", [
    (None, "GET", "Authorization"), ("null", "GET", "Authorization"),
    ("https://evil.example", "POST", "Authorization"),
    (ORIGIN, "PATCH", "Authorization"), (ORIGIN, "GET", "Authorization, X-Admin"),
])
def test_preflight_rejects_untrusted_origins_methods_and_headers(api_server, origin, method, requested_headers):
    headers = {"Access-Control-Request-Method": method, "Access-Control-Request-Headers": requested_headers}
    if origin is not None:
        headers["Origin"] = origin
    status, returned_headers, payload = request(api_server(), "OPTIONS", headers=headers)
    assert status == 403
    assert payload["error"]["code"] in {"cross_origin_forbidden", "preflight_forbidden"}
    assert "Access-Control-Allow-Methods" not in returned_headers


def test_rejected_write_drains_body_and_preserves_windows_json_error(api_server):
    server = api_server()
    body = "x" * (1024 * 1024)
    for headers, expected in (({"Origin": ORIGIN}, 401), ({"Origin": "https://evil.example"}, 403)):
        status, _, payload = request(server, "POST", "/api/validate", headers=headers, body=body)
        assert status == expected
        assert payload["error"]["code"] in {"authentication_required", "cross_origin_forbidden"}
    assert request(server, headers={"Authorization": "Bearer " + TOKEN})[0] == 200


def test_authorized_oversized_json_retains_original_drained_error(api_server, monkeypatch):
    monkeypatch.setattr(web, "JSON_BODY_LIMIT_BYTES", 128)
    status, headers, payload = request(api_server(), "POST", "/api/validate", headers={
        "Authorization": "Bearer " + TOKEN, "Origin": ORIGIN, "Content-Type": "application/json",
    }, body='{"value": "' + "x" * 512 + '"}')
    assert status == 413 and payload["error"]["code"] == "body_too_large"
    assert headers["Access-Control-Allow-Origin"] == ORIGIN


@pytest.mark.parametrize("duplicate,expected", [("Origin", 403), ("Authorization", 401)])
def test_duplicate_security_headers_are_not_accepted(api_server, duplicate, expected):
    server = api_server()
    client = HTTPConnection(*server.server_address[:2], timeout=5)
    try:
        client.putrequest("GET", "/api/health")
        client.putheader("Origin", ORIGIN)
        client.putheader("Authorization", "Bearer " + TOKEN)
        client.putheader(duplicate, ORIGIN if duplicate == "Origin" else "Bearer " + TOKEN)
        client.endheaders()
        response = client.getresponse()
        assert response.status == expected
        assert json.loads(response.read())["error"]["code"] in {"cross_origin_forbidden", "authentication_required"}
        if duplicate == "Origin":
            assert response.getheader("Access-Control-Allow-Origin") is None
    finally:
        client.close()


def test_local_default_keeps_unauthenticated_health_and_same_origin_writes(api_server):
    server = api_server(remote=False)
    status, headers, _ = request(server)
    assert status == 200 and "Access-Control-Allow-Origin" not in headers
    local_origin = "http://{}:{}".format(*server.server_address[:2])
    for origin in (local_origin, None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        assert request(server, "POST", "/api/validate", headers=headers, body="{}")[0] == 200
    status, headers, payload = request(server, "POST", "/api/validate", headers={
        "Origin": ORIGIN, "Content-Type": "application/json",
    }, body="{}")
    assert status == 403 and payload["error"]["code"] == "cross_origin_forbidden"
    assert "Access-Control-Allow-Origin" not in headers
    assert request(server, "POST", "/api/validate", headers={"Sec-Fetch-Site": "cross-site"}, body="{}")[0] == 403


@pytest.mark.parametrize("origin", ["*", "http://zh.gripe", "https://zh.gripe/", "https://user:pass@zh.gripe", "https://zh.gripe/path", "https://zh.gripe?query", "https://zh.gripe\n", "https://zh.gripe:invalid"])
def test_remote_allowlist_rejects_invalid_or_non_https_origins(origin):
    with pytest.raises(ValueError, match="trusted_origins"):
        web.build_server(port=0, remote_access_token=TOKEN, trusted_origins=(origin,))
