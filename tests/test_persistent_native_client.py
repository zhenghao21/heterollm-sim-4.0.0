"""Local protocol tests; no llama.cpp, GPU, or native timing experiments."""
import importlib.util
import json
import threading
from contextlib import contextmanager
from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

import pytest


from tools.native_http_client import PersistentClient


@contextmanager
def fake_server(*, silent_close=False, status=200):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            with self.server.lock:
                self.server.requests.append({"path": self.path, "payload": payload,
                                             "connection": self.connection,
                                             "connection_header": self.headers.get("Connection")})
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Keep-Alive", "timeout=5, max=100")
            self.end_headers()
            chunks = [b'data: {"content":"x","tokens":[42]}\n\n',
                      b'data: {"stop":true,"timings":{"predicted_n":1}}\n\n',
                      b'data: [DONE]\n\n', b': after application terminator\n\n']
            for chunk in chunks:
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.write(b"0\r\nX-Trailer: end\r\n\r\n")
            self.wfile.flush()
            if silent_close:
                # Same externally visible behavior as the locked server:
                # valid chunk termination, keep-alive header, then socket close.
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.requests = []
    server.lock = threading.Lock()
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}/completion?probe=1"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_persistent_body_drained_after_sse_done_and_queries_preserved():
    with fake_server() as (server, url):
        client = PersistentClient(parallel=2)
        try:
            first = client.batch(url, {"stream": True})
            second = client.batch(url, {"stream": True})
            assert client.connect_counts == 2
            assert len(server.requests) == 4
            assert len({id(r["connection"]) for r in server.requests}) == 2
            assert all(r["path"] == "/completion?probe=1" for r in server.requests)
            assert all(r["connection_header"] != "close" for r in server.requests)
            assert all(response["content"] == "x" for response, _ in first + second)
            assert all(b["client_transport"]["connections_created_before_batch_barrier"] == 0
                       for _, b in second)
        finally:
            client.close()


def test_server_silent_close_not_retried_or_reported_as_persistent_success():
    with fake_server(silent_close=True) as (server, url):
        client = PersistentClient(parallel=1)
        try:
            client.batch(url, {"stream": True})
            # Wait for EOF deterministically without consuming it or introducing
            # sleeps/timing assumptions; the next POST must not be replayed.
            connection = client.connections[0]
            import socket
            assert connection.sock.recv(1, socket.MSG_PEEK) == b""
            with pytest.raises((RemoteDisconnected, ConnectionResetError,
                                ConnectionAbortedError, BrokenPipeError)):
                client.batch(url, {"stream": True})
            assert len(server.requests) == 1
            assert client.connect_counts == 1
        finally:
            client.close()


def test_explicit_fresh_preconnected_batches_work_when_server_closes_each_response():
    with fake_server(silent_close=True) as (server, url):
        client = PersistentClient(parallel=2, connection_policy="preconnect_each_batch")
        try:
            first = client.batch(url, {"stream": True})
            second = client.batch(url, {"stream": True})
            assert len(server.requests) == 4
            assert client.connect_counts == 4
            assert all(b["client_transport"]["connection_policy"] == "preconnect_each_batch"
                       for _, b in first + second)
            assert all(b["client_transport"]["post_retry_count"] == 0 for _, b in first + second)
            assert all(b["client_transport"]["connections_created_before_batch_barrier"] == 2
                       for _, b in first + second)
        finally:
            client.close()


def test_http_error_not_parsed_as_successful_measurement():
    with fake_server(status=503) as (server, url):
        client = PersistentClient(parallel=1)
        try:
            with pytest.raises(HTTPError) as error:
                client.batch(url, {"stream": True})
            assert error.value.code == 503
            assert len(server.requests) == 1
        finally:
            client.close()


def test_policy_and_lifecycle_are_explicit():
    with pytest.raises(ValueError):
        PersistentClient(parallel=0)
    with pytest.raises(ValueError):
        PersistentClient(connection_policy="retry")
    client = PersistentClient(parallel=1)
    client.close()
    client.close()
    with pytest.raises(RuntimeError):
        client.batch("http://127.0.0.1:1/completion", {})
