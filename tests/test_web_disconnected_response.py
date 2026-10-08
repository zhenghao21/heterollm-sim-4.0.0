"""A disconnected observer must not turn a healthy run into an HTTP fault."""

from types import SimpleNamespace

import pytest

from heterollm_sim.web import HeteroLLMRequestHandler


def polling_handler(snapshot, *, write_failure=None, fail_at=1):
    handler = object.__new__(HeteroLLMRequestHandler)
    handler.command = "GET"
    handler.path = "/api/run-jobs/existing-job"
    handler.request_version = "HTTP/1.1"
    handler.requestline = "GET /api/run-jobs/existing-job HTTP/1.1"
    handler.close_connection = False
    handler.log_request = lambda *_: None
    handler._run_job_manager = lambda: SimpleNamespace(get=lambda _: snapshot)
    handler.faults = []
    handler._send_unexpected_error = handler.faults.append
    handler.writes = []

    def write(data):
        handler.writes.append(data)
        if write_failure is not None and len(handler.writes) == fail_at:
            raise write_failure
        return len(data)

    handler.wfile = SimpleNamespace(write=write)
    return handler


@pytest.mark.parametrize("error_type", [BrokenPipeError, ConnectionResetError, ConnectionAbortedError])
@pytest.mark.parametrize("fail_at", [1, 2], ids=["headers", "body"])
def test_poll_disconnect_closes_connection_without_retry_or_internal_error(error_type, fail_at):
    snapshot = {"job_id": "existing-job", "status": "running", "progress": {"completed": 12}}
    handler = polling_handler(snapshot, write_failure=error_type("client closed"), fail_at=fail_at)
    handler.do_GET()
    assert handler.close_connection is True
    assert handler.faults == []
    assert len(handler.writes) == fail_at  # No second HTTP response on a dead connection.
    assert handler.writes[0].startswith(b"HTTP/1.0 200 ")
    assert snapshot == {"job_id": "existing-job", "status": "running", "progress": {"completed": 12}}


def test_normal_poll_keeps_complete_body_and_headers():
    handler = polling_handler({"job_id": "existing-job", "status": "completed"})
    handler.do_GET()
    assert not handler.close_connection
    assert handler.faults == []
    assert len(handler.writes) == 2
    assert b"Content-Length: " + str(len(handler.writes[1])).encode() in handler.writes[0]
    assert b'"status": "completed"' in handler.writes[1] or b'"status":"completed"' in handler.writes[1]


def test_unrelated_write_failure_is_still_an_internal_error():
    failure = OSError("unexpected local I/O failure")
    handler = polling_handler({"status": "running"}, write_failure=failure, fail_at=2)
    handler.do_GET()
    assert handler.faults == [failure]


def test_backend_connection_failure_is_not_misclassified_as_client_disconnect():
    failure = ConnectionAbortedError("backend dependency failed")
    handler = polling_handler({})

    def get(_):
        raise failure

    handler._run_job_manager = lambda: SimpleNamespace(get=get)
    handler.do_GET()
    assert handler.faults == [failure]
    assert handler.writes == []
