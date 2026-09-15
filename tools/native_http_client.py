"""Bounded diagnostic HTTP clients using the unchanged native SSE extractor.

``persistent`` is strict: a failed POST is never transparently replayed.
``preconnect_each_batch`` creates each socket before the batch barrier; it is
explicitly a fresh-connection experiment, not successful connection reuse.
"""
from __future__ import annotations

import http.client
import io
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.parse import urlsplit

from tools.native_llama_compare import post_stream_json, _client_batch_makespan


class _DrainedResponse:
    """Leave no chunk framing/trailers behind when the parser stops at [DONE]."""

    def __init__(self, response: http.client.HTTPResponse):
        self.response = response

    def __enter__(self):
        return self.response

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                # [DONE] is an SSE marker, not the HTTP body boundary. The
                # locked parser may also stop early for a plain JSON response.
                self.response.read()
        finally:
            self.response.close()
        return False


class PersistentClient:
    def __init__(self, parallel=4, *, connection_policy="persistent", timeout=180):
        if not isinstance(parallel, int) or isinstance(parallel, bool) or parallel < 1:
            raise ValueError("parallel must be a positive integer")
        if connection_policy not in {"persistent", "preconnect_each_batch"}:
            raise ValueError("unsupported connection_policy")
        self.parallel = parallel
        self.connection_policy = connection_policy
        self.timeout = timeout
        self.pool = ThreadPoolExecutor(max_workers=parallel)
        self.local = threading.local()
        self.connections = []
        self.lock = threading.Lock()
        self.batch_lock = threading.Lock()
        self.connect_counts = 0
        self.closed = False
        namespace = dict(post_stream_json.__globals__)
        namespace["urlopen"] = self.open
        self.parse = types.FunctionType(post_stream_json.__code__, namespace)

    def open(self, req, timeout=180):
        connection = self.local.connection
        parsed = urlsplit(req.full_url)
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        # HTTPConnection deliberately does not retry a failed POST. A silent
        # server close must invalidate this run, never create another request.
        connection.request("POST", target, body=req.data, headers=dict(req.header_items()))
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            try:
                body = response.read()
            finally:
                response.close()
            raise HTTPError(req.full_url, response.status, response.reason,
                            response.headers, io.BytesIO(body))
        return _DrainedResponse(response)

    def batch(self, url, payload):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("diagnostic client supports an explicit http URL only")
        origin = (parsed.hostname, parsed.port or 80)
        with self.batch_lock:
            if self.closed:
                raise RuntimeError("client is closed")
            barrier = threading.Barrier(self.parallel)
            before_connects = self.connect_counts

            def one(_):
                try:
                    connection = getattr(self.local, "connection", None)
                    if connection is None or getattr(self.local, "origin", None) != origin:
                        if connection is not None:
                            connection.close()
                        connection = http.client.HTTPConnection(*origin, timeout=self.timeout)
                        self.local.connection = connection
                        self.local.origin = origin
                        with self.lock:
                            self.connections.append(connection)
                    if self.connection_policy == "preconnect_each_batch":
                        connection.close()
                    if connection.sock is None:
                        connection.connect()
                        with self.lock:
                            self.connect_counts += 1
                    barrier.wait(timeout=30)
                    return self.parse(url, payload)
                except BaseException:
                    barrier.abort()
                    raise

            # Await every worker even on error, then propagate the first error.
            # This leaves no outstanding requests racing a later batch/cleanup.
            futures = [self.pool.submit(one, i) for i in range(self.parallel)]
            result, failures = [], []
            for future in futures:
                try:
                    result.append(future.result())
                except BaseException as exc:
                    failures.append(exc)
            if failures:
                raise failures[0]
            makespan = _client_batch_makespan(result)
            transport = {
                "connection_policy": self.connection_policy,
                "connections_created_before_batch_barrier": self.connect_counts - before_connects,
                "connections_created_total": self.connect_counts,
                "post_retry_count": 0,
            }
            for _, boundary in result:
                boundary["batch_client_makespan_ms"] = makespan
                boundary["client_transport"] = dict(transport)
            return result

    def close(self):
        with self.batch_lock:
            if self.closed:
                return
            self.closed = True
            self.pool.shutdown(wait=True)
            for connection in self.connections:
                connection.close()
