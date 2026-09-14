"""Dependency-free local HTTP API and static Web UI server."""

from __future__ import annotations

from dataclasses import dataclass
import json
import mimetypes
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlsplit
import sys
import webbrowser

from . import __version__
from .architecture_presets import (
    architecture_preset_detail,
    architecture_preset_page,
)
from .architecture_scan import scan_architecture_candidates
from .component_presets import component_preset_detail, component_preset_page
from .compiler_ir import compile_canonical_scenario
from .config import ScenarioConfig, scenario_from_dict
from .contracts import (
    DEFAULT_VISUALIZATION_EVENT_LIMIT,
    DEFAULT_VISUALIZATION_MEMORY_SEGMENT_LIMIT,
    MAX_VISUALIZATION_EVENT_LIMIT,
    MAX_VISUALIZATION_MEMORY_SEGMENT_LIMIT,
)
from .control_plane_state import mapping_fingerprint_status
from .model_catalog import CatalogError, HuggingFaceClient, ModelCatalog
from .planner import validate_scenario
from .protocol_presets import protocol_preset_detail, protocol_preset_page
from .reference import build_reference_scenario
from .reporting import compare_with_gpu_baseline, report_dict, run_scenario
from .run_estimation import estimate_scenario
from .run_jobs import RunJobCapacityError, RunJobManager, RunJobTraceError
from .runtime_diagnostics import (
    diagnostic_log_path,
    health_payload,
    record_unexpected_exception,
)
from .serde import canonical_json, to_primitive
from .topology import ValidationIssue, validate_topology


# Explicit placement/control-plane metadata can legitimately exceed the old
# 1 MiB boundary while still fitting in the browser's local scenario store.
JSON_BODY_LIMIT_BYTES = 8 * 1024 * 1024
JSON_BODY_DRAIN_CHUNK_BYTES = 64 * 1024
JSON_BODY_DRAIN_TIMEOUT_S = 2.0
STATIC_DIR = "webui"
FALLBACK_INDEX = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HeteroLLM Simulator</title>
</head>
<body>
  <main>
    <h1>HeteroLLM Simulator</h1>
    <p>此构建未包含 Web UI 资源。</p>
  </main>
</body>
</html>
""".encode("utf-8")


@dataclass(frozen=True)
class HttpError(Exception):
    status: int
    code: str
    message: str
    details: Optional[Any] = None
    message_en: Optional[str] = None
    diagnostic_id: Optional[str] = None
    exception_type: Optional[str] = None


class HeteroLLMRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the local simulator API."""

    server_version = "HeteroLLMSim/{}".format(__version__)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API name
        split = urlsplit(self.path)
        path = split.path
        try:
            if path == "/api/health":
                self._send_json(200, health_payload(__version__))
                return
            if path == "/api/reference":
                self._send_json(200, scenario_to_payload(build_reference_scenario()))
                return
            if path == "/api/model-presets":
                params = parse_qs(split.query, keep_blank_values=True)
                catalog = self._model_catalog()
                allowed_keys = {
                    "limit",
                    "offset",
                    "query",
                    "family",
                    "architecture",
                    "model_kind",
                    "support_level",
                    "openness",
                    "source",
                }
                unknown_keys = sorted(set(params) - allowed_keys)
                if unknown_keys:
                    raise HttpError(
                        400,
                        "unknown_query_fields",
                        "模型预设查询包含未知字段：{}".format(", ".join(unknown_keys)),
                        message_en="unknown model-preset query fields: {}".format(", ".join(unknown_keys)),
                    )
                self._send_json(
                    200,
                    catalog.page(
                        offset=_query_integer(params, "offset", 0),
                        limit=_query_integer(params, "limit", 50),
                        query=_query_value(params, "query"),
                        family=_query_value(params, "family"),
                        architecture=_query_value(params, "architecture"),
                        model_kind=_query_value(params, "model_kind"),
                        support_level=_query_value(params, "support_level"),
                        openness=_query_value(params, "openness"),
                        source=_query_value(params, "source"),
                    ),
                )
                return
            if path == "/api/model-presets/remote-search":
                params = parse_qs(split.query, keep_blank_values=True)
                query = _query_value(params, "query")
                limit = _query_integer(params, "limit", 20)
                items = self._model_catalog().remote_search(query, limit)
                self._send_json(
                    200,
                    {
                        "items": items,
                        "total": len(items),
                        "query": query,
                        "limit": limit,
                    },
                )
                return
            if path.startswith("/api/model-presets/"):
                if path == "/api/model-presets/import":
                    raise HttpError(405, "method_not_allowed", "此端点要求使用 POST 方法", message_en="endpoint requires POST")
                preset_id = unquote(path[len("/api/model-presets/") :])
                try:
                    detail = self._model_catalog().detail(preset_id)
                except KeyError as exc:
                    raise HttpError(
                        404,
                        "not_found",
                        "未找到指定的模型预设",
                        message_en="unknown model preset",
                    ) from exc
                self._send_json(200, detail)
                return
            if path == "/api/component-presets":
                params = parse_qs(split.query, keep_blank_values=True)
                component_kind = (
                    _query_value(params, "component_kind")
                    or _query_value(params, "kind")
                    or ""
                )
                self._send_json(
                    200,
                    component_preset_page(
                        query=_query_value(params, "query") or "",
                        family=_query_value(params, "family") or "",
                        component_kind=component_kind,
                        evidence_level=_query_value(params, "evidence_level") or "",
                        tag=_query_value(params, "tag") or "",
                    ),
                )
                return
            if path.startswith("/api/component-presets/"):
                preset_id = unquote(path[len("/api/component-presets/") :])
                try:
                    detail = component_preset_detail(preset_id)
                except KeyError as exc:
                    raise HttpError(
                        404,
                        "not_found",
                        "未找到指定的组件预设",
                        message_en="unknown component preset",
                    ) from exc
                self._send_json(200, detail)
                return
            if path == "/api/architecture-presets":
                params = parse_qs(split.query, keep_blank_values=True)
                loadable_value = _query_value(params, "loadable")
                loadable = None
                if loadable_value not in {None, ""}:
                    normalized_loadable = loadable_value.strip().lower()
                    if normalized_loadable in {"true", "1"}:
                        loadable = True
                    elif normalized_loadable in {"false", "0"}:
                        loadable = False
                    else:
                        raise HttpError(
                            400,
                            "invalid_loadable_filter",
                            "loadable 筛选值必须为 true、false、1 或 0",
                            message_en="loadable filter must be true, false, 1, or 0",
                        )
                self._send_json(
                    200,
                    architecture_preset_page(
                        query=_query_value(params, "query") or "",
                        vendor=_query_value(params, "vendor") or "",
                        family=_query_value(params, "family") or "",
                        topology_class=_query_value(params, "topology_class") or "",
                        scale=_query_value(params, "scale") or "",
                        support_level=_query_value(params, "support_level") or "",
                        protocol=_query_value(params, "protocol") or "",
                        tag=_query_value(params, "tag") or "",
                        loadable=loadable,
                    ),
                )
                return
            if path.startswith("/api/architecture-presets/"):
                preset_id = unquote(path[len("/api/architecture-presets/") :])
                if not preset_id or "/" in preset_id:
                    raise HttpError(
                        404,
                        "not_found",
                        "未找到指定的架构拓扑预设",
                        message_en="unknown architecture topology preset",
                    )
                try:
                    detail = architecture_preset_detail(preset_id)
                except KeyError as exc:
                    raise HttpError(
                        404,
                        "not_found",
                        "未找到指定的架构拓扑预设",
                        message_en="unknown architecture topology preset",
                    ) from exc
                self._send_json(200, detail)
                return
            if path == "/api/protocol-presets":
                params = parse_qs(split.query, keep_blank_values=True)
                self._send_json(
                    200,
                    protocol_preset_page(
                        query=_query_value(params, "query") or "",
                        protocol=_query_value(params, "protocol") or "",
                        organization=_query_value(params, "organization") or "",
                    ),
                )
                return
            if path.startswith("/api/protocol-presets/"):
                preset_id = unquote(path[len("/api/protocol-presets/") :])
                try:
                    detail = protocol_preset_detail(preset_id)
                except KeyError as exc:
                    raise HttpError(
                        404,
                        "not_found",
                        "未找到指定的通信协议预设",
                        message_en="unknown communication protocol preset",
                    ) from exc
                self._send_json(200, detail)
                return
            if path.startswith("/api/run-jobs/") and path.endswith("/trace"):
                job_id = unquote(
                    path[len("/api/run-jobs/") : -len("/trace")]
                ).strip("/")
                if not job_id or "/" in job_id:
                    raise HttpError(
                        404,
                        "run_job_not_found",
                        "未找到指定的仿真任务",
                        message_en="simulation job not found",
                    )
                batch_id, offset, limit = _run_job_trace_query(split.query)
                try:
                    page = self._run_job_manager().trace_page(
                        job_id,
                        batch_id,
                        offset=offset,
                        limit=limit,
                    )
                except RunJobTraceError as exc:
                    raise HttpError(
                        exc.status,
                        exc.code,
                        exc.message_zh,
                        exc.details or None,
                        exc.message_en,
                    ) from exc
                self._send_json(200, page)
                return
            if path.startswith("/api/run-jobs/"):
                if path.endswith("/cancel"):
                    raise HttpError(
                        405,
                        "method_not_allowed",
                        "取消仿真任务要求使用 POST 方法",
                        message_en="cancelling a simulation job requires POST",
                    )
                suffix = unquote(path[len("/api/run-jobs/") :]).strip("/")
                if not suffix or "/" in suffix:
                    raise HttpError(
                        404,
                        "run_job_not_found",
                        "未找到指定的仿真任务",
                        message_en="simulation job not found",
                    )
                snapshot = self._run_job_manager().get(suffix)
                if snapshot is None:
                    raise HttpError(
                        404,
                        "run_job_not_found",
                        "未找到指定的仿真任务",
                        message_en="simulation job not found",
                    )
                self._send_json(200, snapshot)
                return
            if path in {
                "/api/validate",
                "/api/run",
                "/api/run-estimate",
                "/api/run-jobs",
                "/api/canonical-ir",
                "/api/architecture-scan",
                "/api/compare",
            }:
                raise HttpError(
                    405,
                    "method_not_allowed",
                    "此端点要求使用 POST 方法",
                    message_en="endpoint requires POST",
                )
            if path.startswith("/api/"):
                raise HttpError(404, "not_found", "未知的 API 端点", message_en="unknown API endpoint")
            self._send_static(path)
        except HttpError as exc:
            self._send_error(exc)
        except CatalogError as exc:
            self._send_error(
                HttpError(
                    exc.status,
                    exc.code,
                    "模型目录请求失败",
                    message_en=str(exc),
                )
            )
        except Exception as exc:
            self._send_unexpected_error(exc)

    def do_POST(self) -> None:  # noqa: N802 - stdlib API name
        split = urlsplit(self.path)
        path = split.path
        try:
            self._require_same_origin()
            if path == "/api/validate":
                payload = self._read_json_object()
                self._send_json(200, validation_payload(payload))
                return
            if path == "/api/run":
                payload = self._read_json_object()
                scenario = scenario_or_http_error(payload)
                ensure_valid_or_http_error(scenario)
                trace_options = _visualization_query_options(split.query)
                self._send_json(
                    200,
                    report_dict(run_scenario(scenario), **trace_options),
                )
                return
            if path == "/api/run-estimate":
                payload = self._read_json_object()
                scenario = scenario_or_http_error(payload)
                ensure_valid_or_http_error(scenario)
                self._send_json(200, estimate_scenario(scenario))
                return
            if path == "/api/run-jobs":
                payload = self._read_json_object()
                scenario, retention_policy = _run_job_request(payload)
                ensure_valid_or_http_error(scenario)
                try:
                    job_id = self._run_job_manager().submit(
                        scenario,
                        retention_policy=retention_policy,
                    )
                except RunJobCapacityError as exc:
                    raise HttpError(
                        429,
                        "run_job_capacity_reached",
                        str(exc),
                        message_en="background simulation queue is full",
                    ) from exc
                snapshot = self._run_job_manager().get(job_id)
                self._send_json(202, snapshot)
                return
            if path == "/api/canonical-ir":
                payload = self._read_json_object()
                scenario = scenario_or_http_error(payload)
                ensure_valid_or_http_error(scenario)
                self._send_json(
                    200,
                    to_primitive(compile_canonical_scenario(scenario)),
                )
                return
            if path == "/api/architecture-scan":
                payload = self._read_json_object()
                scenario, backend, top_n = _architecture_scan_request(payload)
                ensure_valid_or_http_error(scenario)
                self._send_json(
                    200,
                    scan_architecture_candidates(
                        scenario,
                        backend=backend,
                        top_n=top_n,
                    ),
                )
                return
            if path.startswith("/api/run-jobs/") and path.endswith("/cancel"):
                job_id = unquote(
                    path[len("/api/run-jobs/") : -len("/cancel")]
                ).strip("/")
                manager = self._run_job_manager()
                snapshot = manager.get(job_id)
                if snapshot is None:
                    raise HttpError(
                        404,
                        "run_job_not_found",
                        "未找到指定的仿真任务",
                        message_en="simulation job not found",
                    )
                if not manager.cancel(job_id):
                    raise HttpError(
                        409,
                        "run_job_not_cancellable",
                        "仿真任务已经结束，无法取消",
                        details={"status": snapshot["status"]},
                        message_en="simulation job is already terminal",
                    )
                self._send_json(202, manager.get(job_id))
                return
            if path == "/api/compare":
                payload = self._read_json_object()
                scenario = scenario_or_http_error(payload)
                ensure_valid_or_http_error(scenario)
                self._send_json(200, compare_with_gpu_baseline(scenario))
                return
            if path == "/api/model-presets/import":
                payload = self._read_json_object()
                unexpected = sorted(set(payload) - {"repo_id", "revision"})
                if unexpected:
                    raise HttpError(
                        400,
                        "unknown_fields",
                        "导入请求包含未知字段：{}".format(", ".join(unexpected)),
                        message_en="unknown import fields: {}".format(", ".join(unexpected)),
                    )
                if "repo_id" not in payload:
                    raise HttpError(400, "missing_repo_id", "缺少必填字段 repo_id", message_en="repo_id is required")
                detail = self._model_catalog().import_repository(
                    payload["repo_id"], payload.get("revision", "main")
                )
                self._send_json(200, detail)
                return
            if (
                path
                in {
                    "/api/health",
                    "/api/reference",
                    "/api/model-presets",
                    "/api/component-presets",
                    "/api/architecture-presets",
                    "/api/protocol-presets",
                }
                or path.startswith("/api/model-presets/")
                or path.startswith("/api/component-presets/")
                or path.startswith("/api/architecture-presets/")
                or path.startswith("/api/protocol-presets/")
            ):
                raise HttpError(
                    405,
                    "method_not_allowed",
                    "此端点要求使用 GET 方法",
                    message_en="endpoint requires GET",
                )
            if path.startswith("/api/"):
                raise HttpError(404, "not_found", "未知的 API 端点", message_en="unknown API endpoint")
            raise HttpError(404, "not_found", "未知端点", message_en="unknown endpoint")
        except HttpError as exc:
            self._send_error(exc)
        except CatalogError as exc:
            self._send_error(
                HttpError(
                    exc.status,
                    exc.code,
                    "模型目录请求失败",
                    message_en=str(exc),
                )
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._send_error(
                HttpError(
                    422,
                    "scenario_error",
                    str(exc) if re.search(r"[\u3400-\u9fff]", str(exc)) else "场景处理失败；请检查输入配置",
                    message_en=None if re.search(r"[\u3400-\u9fff]", str(exc)) else str(exc),
                )
            )
        except Exception as exc:
            self._send_unexpected_error(exc)

    def log_message(self, format: str, *args: object) -> None:
        """Keep local test and CLI output focused on explicit startup messages."""

    def _read_json_object(self) -> Dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            raise HttpError(
                415,
                "unsupported_media_type",
                "Content-Type 必须为 application/json",
                message_en="Content-Type must be application/json",
            )
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise HttpError(
                411,
                "content_length_required",
                "缺少必需的 Content-Length 请求头",
                message_en="Content-Length header is required",
            )
        try:
            length = int(raw_length)
        except ValueError:
            raise HttpError(400, "bad_content_length", "Content-Length 无效", message_en="Content-Length is invalid")
        if length > JSON_BODY_LIMIT_BYTES:
            # On Windows, closing a socket while request bytes are still unread
            # commonly turns the intended HTTP 413 into ECONNRESET in fetch().
            # Drain in bounded-memory chunks so the browser receives the JSON
            # error envelope and does not incorrectly mark the whole API down.
            self._drain_rejected_body(length)
            raise HttpError(
                413,
                "body_too_large",
                "JSON 请求体超过 {} 字节限制".format(JSON_BODY_LIMIT_BYTES),
                details={
                    "actual_bytes": length,
                    "limit_bytes": JSON_BODY_LIMIT_BYTES,
                },
                message_en="JSON body exceeds {} bytes".format(JSON_BODY_LIMIT_BYTES),
            )
        if length <= 0:
            raise HttpError(400, "bad_json", "请求体必须是有效的 JSON", message_en="request body must be valid JSON")
        raw_body = self.rfile.read(length)
        if len(raw_body) != length:
            raise HttpError(400, "bad_request", "请求体未完整传输", message_en="request body ended early")
        try:
            def reject_non_finite(token: str) -> None:
                raise ValueError("JSON 不允许非有限数值：{}".format(token))

            payload = json.loads(
                raw_body.decode("utf-8"),
                parse_constant=reject_non_finite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise HttpError(400, "bad_json", "请求体必须是有效的 JSON", message_en="request body must be valid JSON")
        if not isinstance(payload, dict):
            raise HttpError(
                400,
                "bad_json_object",
                "JSON 顶层值必须是对象",
                message_en="top-level JSON value must be an object",
            )
        return payload

    def _drain_rejected_body(self, length: int) -> bool:
        """Consume a rejected body without buffering it, before a short deadline."""

        remaining = max(0, length)
        deadline = time.monotonic() + JSON_BODY_DRAIN_TIMEOUT_S
        previous_timeout = self.connection.gettimeout()
        try:
            while remaining:
                seconds_left = deadline - time.monotonic()
                if seconds_left <= 0:
                    break
                self.connection.settimeout(seconds_left)
                chunk = self.rfile.read(min(remaining, JSON_BODY_DRAIN_CHUNK_BYTES))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            self.close_connection = True
        finally:
            try:
                self.connection.settimeout(previous_timeout)
            except OSError:
                self.close_connection = True
        if remaining:
            self.close_connection = True
        return remaining == 0

    def _model_catalog(self) -> ModelCatalog:
        catalog = getattr(self.server, "model_catalog", None)
        if not isinstance(catalog, ModelCatalog):
            raise HttpError(500, "catalog_unavailable", "模型目录当前不可用", message_en="model catalog is unavailable")
        return catalog

    def _run_job_manager(self) -> RunJobManager:
        manager = getattr(self.server, "run_job_manager", None)
        if not isinstance(manager, RunJobManager):
            raise HttpError(
                500,
                "run_jobs_unavailable",
                "后台仿真任务服务当前不可用",
                message_en="background simulation job service is unavailable",
            )
        return manager

    def _require_same_origin(self) -> None:
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        if origin and (not host or origin.rstrip("/") != "http://{}".format(host)):
            self._reject_cross_origin_request()
        if self.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
            self._reject_cross_origin_request()

    def _reject_cross_origin_request(self) -> None:
        """Drain a declared POST body so Windows can deliver the 403 envelope."""

        raw_length = self.headers.get("Content-Length")
        if raw_length is not None:
            try:
                length = int(raw_length)
            except ValueError:
                self.close_connection = True
            else:
                if length > 0:
                    self._drain_rejected_body(length)
        raise HttpError(
            403,
            "cross_origin_forbidden",
            "不允许跨源 API 请求",
            message_en="cross-origin API requests are not allowed",
        )

    def _send_static(self, request_path: str) -> None:
        asset = static_asset(request_path)
        if asset is None:
            raise HttpError(404, "not_found", "未找到静态资源", message_en="static asset not found")
        body, content_type = asset
        self._send_bytes(200, body, content_type)

    def _send_json(self, status: int, payload: Any) -> None:
        body = (canonical_json(payload) + "\n").encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_error(self, error: HttpError) -> None:
        payload: Dict[str, Any] = {
            "error": {
                "code": error.code,
                "message": error.message,
                "message_zh": error.message,
            }
        }
        if error.message_en:
            payload["error"]["message_en"] = error.message_en
        if error.details is not None:
            payload["error"]["details"] = error.details
        if error.diagnostic_id is not None:
            payload["error"]["diagnostic_id"] = error.diagnostic_id
        if error.exception_type is not None:
            payload["error"]["exception_type"] = error.exception_type
        self._send_json(error.status, payload)

    def _send_unexpected_error(self, exc: BaseException) -> None:
        diagnostic_id = record_unexpected_exception(
            exc,
            context="{} {}".format(self.command, urlsplit(self.path).path),
            log_path=getattr(self.server, "diagnostic_log_path", None),
        )
        self._send_error(
            HttpError(
                500,
                "internal_error",
                "服务器发生未预料的异常；请记录诊断 ID 并查看本地诊断日志",
                message_en="internal server error",
                diagnostic_id=diagnostic_id,
                exception_type=type(exc).__name__,
            )
        )

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'",
        )
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def validation_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        scenario = scenario_from_dict(payload)
    except (ValueError, TypeError, KeyError) as exc:
        message_zh, message_en = _scenario_parse_messages(exc)
        return {
            "valid": False,
            "errors": {
                "topology": [],
                "scenario": [
                    _message_issue(
                        "parse_error",
                        message_zh,
                        message_en,
                    )
                ],
            },
            "warnings": {"topology": [], "scenario": []},
            "information": {"topology": [], "scenario": []},
        }
    return validation_payload_for_scenario(scenario)


def _scenario_parse_messages(exc: BaseException) -> Tuple[str, str]:
    """Return an actionable Chinese parse error while retaining diagnostics.

    Configuration objects still use stable English field identifiers.  The
    user-facing sentence is Chinese and points at the actual failing path;
    the original exception remains available as ``message_en`` for logs and
    machine consumers.
    """

    detail = str(exc).strip() or type(exc).__name__
    if re.search(r"[\u3400-\u9fff]", detail):
        return "场景 JSON 解析失败：{}".format(detail), detail

    patterns = (
        (
            r"rank_mapping must contain exactly world_size entries",
            lambda match: "placement.parallel.rank_mapping 的条目数必须与 world_size 完全一致",
        ),
        (
            r"rank_mapping ranks must cover \[0, world_size\)",
            lambda match: "placement.parallel.rank_mapping 的 rank 必须完整覆盖 [0, world_size)",
        ),
        (
            r"parallel rank coordinates must be unique",
            lambda match: "placement.parallel.rank_mapping 中的 TP/PP/EP 坐标必须唯一",
        ),
        (
            r"(tp|pp|ep)_rank must be less than (tp|pp|ep)_degree",
            lambda match: "placement.parallel.rank_mapping 中的 {}_rank 必须小于 {}_degree".format(
                match.group(1), match.group(2)
            ),
        ),
        (
            r"rank_mapping must contain exactly (\d+) ranks",
            lambda match: "placement.parallel.rank_mapping 必须恰好包含 {} 个 rank".format(
                match.group(1)
            ),
        ),
        (
            r"rank_mapping rank values must be unique",
            lambda match: "placement.parallel.rank_mapping 中的 rank 值必须唯一",
        ),
        (
            r"rank_mapping must cover every TP/PP/EP coordinate exactly once",
            lambda match: "placement.parallel.rank_mapping 必须且只能覆盖每个 TP/PP/EP 坐标一次",
        ),
        (
            r"(.+) conflicts with parallel\.(.+)",
            lambda match: "placement.{} 与 placement.parallel.{} 冲突".format(
                match.group(1), match.group(2)
            ),
        ),
        (
            r"(.+) must be an object",
            lambda match: "字段 {} 必须是 JSON 对象".format(match.group(1)),
        ),
        (
            r"(.+) must be an array",
            lambda match: "字段 {} 必须是 JSON 数组".format(match.group(1)),
        ),
        (
            r"(.+) must be an integer",
            lambda match: "字段 {} 必须是整数".format(match.group(1)),
        ),
        (
            r"(.+) must be a number",
            lambda match: "字段 {} 必须是数值".format(match.group(1)),
        ),
        (
            r"(.+) must be boolean",
            lambda match: "字段 {} 必须是布尔值".format(match.group(1)),
        ),
    )
    for pattern, render in patterns:
        matched = re.fullmatch(pattern, detail, flags=re.DOTALL)
        if matched is not None:
            return "场景 JSON 解析失败：{}".format(render(matched)), detail
    return "场景 JSON 解析失败：字段值或结构不符合配置约束", detail


def scenario_to_payload(scenario: ScenarioConfig) -> Dict[str, Any]:
    profiles: Dict[str, Any] = {
        "components": to_primitive(scenario.component_profiles),
        "host_orchestration": to_primitive(scenario.host_orchestration_profile),
        "fusion": to_primitive(scenario.fusion_policy),
    }
    if scenario.llama_cpp_config is not None:
        profiles["llama_cpp"] = to_primitive(scenario.llama_cpp_config)
    if scenario.host_output_contract is not None:
        profiles["host_output"] = to_primitive(scenario.host_output_contract)
    if scenario.sampling_policy is not None:
        profiles["sampling"] = to_primitive(scenario.sampling_policy)
    if scenario.cim_interconnect is not None:
        profiles["cim_interconnect"] = to_primitive(scenario.cim_interconnect)
    return {
        "schema_version": scenario.schema_version,
        "name": scenario.name,
        "weights_resident": scenario.weights_resident,
        "assumptions": list(scenario.assumptions),
        "hardware": to_primitive(scenario.hardware),
        "model": to_primitive(scenario.model),
        "placement": to_primitive(scenario.placement),
        "workload": to_primitive(scenario.workload),
        "profiles": profiles,
    }


def validation_payload_for_scenario(scenario: ScenarioConfig) -> Dict[str, Any]:
    topology = validate_topology(scenario.hardware)
    scenario_report = validate_scenario(scenario)
    mapping_status = mapping_fingerprint_status(scenario)
    diagnostics = [dict(item) for item in scenario_report.diagnostics]
    diagnostics_by_message = {
        str(item.get("message_en")): item
        for item in diagnostics
        if item.get("message_en")
    }
    scenario_errors = [
        _message_issue(
            "validation_error",
            message_zh,
            message_en,
            diagnostics_by_message.get(message_en),
        )
        for message_zh, message_en in zip(
            scenario_report.errors, scenario_report.errors_en
        )
        if not message_en.startswith("topology validation failed:")
        and not message_zh.startswith("拓扑校验失败")
    ]
    return {
        "valid": topology.is_valid and scenario_report.is_valid,
        "fingerprint_algorithm": mapping_status["fingerprint_algorithm"],
        "input_fingerprint": mapping_status["input_fingerprint"],
        "current_input_fingerprint": mapping_status[
            "current_input_fingerprint"
        ],
        "mapping_stale": mapping_status["mapping_stale"],
        "mapping": mapping_status,
        "diagnostics": diagnostics,
        "errors": {
            "topology": [_topology_issue(issue) for issue in topology.errors],
            "scenario": scenario_errors,
        },
        "warnings": {
            "topology": [_topology_issue(issue) for issue in topology.warnings],
            "scenario": [
                _message_issue("validation_warning", message_zh, message_en)
                for message_zh, message_en in zip(
                    scenario_report.warnings, scenario_report.warnings_en
                )
            ],
        },
        "information": {
            "topology": [],
            "scenario": [
                _message_issue("validation_information", message_zh, message_en)
                for message_zh, message_en in zip(
                    scenario_report.information, scenario_report.information_en
                )
            ],
        },
    }


def scenario_or_http_error(payload: Mapping[str, Any]) -> ScenarioConfig:
    try:
        return scenario_from_dict(payload)
    except (ValueError, TypeError, KeyError) as exc:
        details = validation_payload(payload)
        message_zh, message_en = _scenario_parse_messages(exc)
        raise HttpError(
            422,
            "invalid_scenario",
            message_zh,
            details,
            message_en,
        ) from exc


def _run_job_request(
    payload: Mapping[str, Any],
) -> Tuple[ScenarioConfig, Optional[str]]:
    unexpected = sorted(set(payload) - {"scenario", "retention_policy"})
    if unexpected:
        raise HttpError(
            400,
            "unknown_fields",
            "后台仿真请求包含未知字段：{}".format(", ".join(unexpected)),
            message_en="unknown background run fields: {}".format(
                ", ".join(unexpected)
            ),
        )
    scenario_payload = payload.get("scenario")
    if not isinstance(scenario_payload, Mapping):
        raise HttpError(
            400,
            "missing_scenario",
            "后台仿真请求缺少有效的 scenario 对象",
            message_en="background run request requires a scenario object",
        )
    retention_policy = payload.get("retention_policy")
    if retention_policy is not None and (
        not isinstance(retention_policy, str) or retention_policy not in {
        "exact",
        "streaming",
        "aggregate",
        }
    ):
        raise HttpError(
            400,
            "invalid_retention_policy",
            "retention_policy 必须为 exact、streaming 或 aggregate",
            message_en=(
                "retention_policy must be exact, streaming, or aggregate"
            ),
        )
    scenario = scenario_or_http_error(scenario_payload)
    scheduler = getattr(scenario.workload, "scheduler", None)
    scheduler_mode = (
        str(getattr(scheduler, "mode", "static")) if scheduler else "static"
    )
    if scheduler_mode == "continuous" and retention_policy not in {
        None,
        "aggregate",
    }:
        raise HttpError(
            400,
            "invalid_retention_policy",
            "continuous 调度仅支持 aggregate retention_policy；"
            "批次外层不会伪装成 exact 或 streaming 任务轨迹",
            message_en=(
                "continuous scheduling supports only aggregate retention_policy; "
                "batch envelopes are not exact or streaming task traces"
            ),
        )
    return scenario, retention_policy


def _architecture_scan_request(
    payload: Mapping[str, Any],
) -> Tuple[ScenarioConfig, str, int]:
    unexpected = sorted(set(payload) - {"scenario", "backend", "top_n"})
    if unexpected:
        raise HttpError(
            400,
            "unknown_fields",
            "架构候选扫描请求包含未知字段：{}".format(", ".join(unexpected)),
            message_en="unknown architecture scan fields: {}".format(
                ", ".join(unexpected)
            ),
        )
    scenario_payload = payload.get("scenario")
    if not isinstance(scenario_payload, Mapping):
        raise HttpError(
            400,
            "missing_scenario",
            "架构候选扫描请求缺少有效的 scenario 对象",
            message_en="architecture scan request requires a scenario object",
        )
    backend = payload.get("backend", "auto")
    if backend not in {"auto", "numpy", "cupy"}:
        raise HttpError(
            400,
            "invalid_backend",
            "架构候选扫描 backend 必须为 auto、numpy 或 cupy",
            message_en="architecture scan backend must be auto, numpy, or cupy",
        )
    top_n = payload.get("top_n", 20)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 200:
        raise HttpError(
            400,
            "invalid_top_n",
            "top_n 必须是 1 到 200 之间的整数",
            message_en="top_n must be an integer between 1 and 200",
        )
    return scenario_or_http_error(scenario_payload), str(backend), top_n


def ensure_valid_or_http_error(scenario: ScenarioConfig) -> None:
    payload = validation_payload_for_scenario(scenario)
    if not payload["valid"]:
        raise HttpError(
            422,
            "invalid_scenario",
            "场景校验失败",
            payload,
            "scenario validation failed",
        )


def static_asset(request_path: str) -> Optional[Tuple[bytes, str]]:
    relative_path = _static_relative_path(request_path)
    if relative_path is None:
        return None
    root = resources.files(__package__).joinpath(STATIC_DIR)
    asset = root.joinpath(*relative_path.split("/"))
    if asset.is_file():
        return asset.read_bytes(), _content_type(relative_path)
    if relative_path == "index.html":
        return FALLBACK_INDEX, "text/html; charset=utf-8"
    return None


class HeteroLLMThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def server_close(self) -> None:
        manager = getattr(self, "run_job_manager", None)
        if isinstance(manager, RunJobManager):
            manager.shutdown(
                wait=False,
                cancel_pending=True,
                cancel_running=True,
            )
        super().server_close()


def build_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    catalog_cache_dir: Optional[Any] = None,
    hf_client: Optional[HuggingFaceClient] = None,
    model_catalog: Optional[ModelCatalog] = None,
    diagnostic_log_path: Optional[Any] = None,
    run_job_manager: Optional[RunJobManager] = None,
) -> ThreadingHTTPServer:
    server = HeteroLLMThreadingHTTPServer((host, port), HeteroLLMRequestHandler)
    server.model_catalog = model_catalog or ModelCatalog(  # type: ignore[attr-defined]
        catalog_cache_dir,
        hf_client=hf_client,
    )
    server.diagnostic_log_path = diagnostic_log_path  # type: ignore[attr-defined]
    server.run_job_manager = run_job_manager or RunJobManager(  # type: ignore[attr-defined]
        max_workers=2,
        max_history=100,
    )
    return server


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
) -> None:
    server = build_server(host, port)
    actual_host, actual_port = server.server_address[:2]
    url = "http://{}:{}/".format(_url_host(actual_host), actual_port)
    print("HeteroLLM Simulator UI 正在监听 {}".format(url), flush=True)
    print("诊断日志：{}".format(diagnostic_log_path()), flush=True)
    print("按 Ctrl+C 停止服务。", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("服务已停止。", file=sys.stderr)
    finally:
        server.server_close()


def _static_relative_path(request_path: str) -> Optional[str]:
    path = unquote(request_path)
    if path in {"", "/"}:
        return "index.html"
    if path.endswith("/"):
        path += "index.html"
    parts = []
    for part in path.lstrip("/").split("/"):
        if part in {"", ".", ".."} or "\\" in part:
            return None
        parts.append(part)
    return "/".join(parts)


def _content_type(relative_path: str) -> str:
    guessed, _ = mimetypes.guess_type(relative_path)
    content_type = guessed or "application/octet-stream"
    if (
        content_type.startswith("text/")
        or content_type in {"application/javascript", "application/json"}
        or content_type == "image/svg+xml"
    ):
        return content_type + "; charset=utf-8"
    return content_type


def _topology_issue(issue: ValidationIssue) -> Dict[str, Any]:
    result = {
        "code": issue.code,
        "message": issue.message,
        "message_zh": issue.message,
    }
    if issue.message_en:
        result["message_en"] = issue.message_en
    if issue.component_id is not None:
        result["component_id"] = issue.component_id
    if issue.port_id is not None:
        result["port_id"] = issue.port_id
    if issue.link_id is not None:
        result["link_id"] = issue.link_id
    return result


def _message_issue(
    code: str,
    message_zh: str,
    message_en: Optional[str] = None,
    diagnostic: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "code": code,
        "message": message_zh,
        "message_zh": message_zh,
    }
    if message_en:
        result["message_en"] = message_en
    if diagnostic:
        structured = dict(diagnostic)
        result["details"] = structured
        for key in (
            "operator_id",
            "sub_operator_id",
            "operator_class",
            "requested_target",
            "resolved_target",
            "resolution_applied",
            "requested_mapping_key",
            "rank_id",
        ):
            if key in structured:
                result[key] = structured[key]
    return result


def _query_value(params: Mapping[str, Any], name: str) -> Optional[str]:
    values = params.get(name)
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != 1:
        raise HttpError(
            400,
            "invalid_query_parameter",
            "查询参数 {} 最多只能提供一次".format(name),
            message_en="{} must be provided at most once".format(name),
        )
    return str(values[0])


def _query_integer(params: Mapping[str, Any], name: str, default: int) -> int:
    value = _query_value(params, name)
    if value in {None, ""}:
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HttpError(
            400,
            "invalid_query_parameter",
            "查询参数 {} 必须是整数".format(name),
            message_en="{} must be an integer".format(name),
        ) from exc


def _run_job_trace_query(query: str) -> Tuple[str, int, int]:
    params = parse_qs(query, keep_blank_values=True)
    unexpected = sorted(set(params) - {"batch_id", "offset", "limit"})
    if unexpected:
        raise HttpError(
            400,
            "invalid_trace_query",
            "批次时间轴请求包含未知查询参数：{}".format(
                ", ".join(unexpected)
            ),
            message_en="unknown batch trace query parameters: {}".format(
                ", ".join(unexpected)
            ),
        )
    batch_id = _query_value(params, "batch_id")
    if batch_id is None or not batch_id.strip():
        raise HttpError(
            400,
            "invalid_batch_id",
            "batch_id 必须是非空字符串",
            message_en="batch_id must be a non-empty string",
        )
    offset = _query_integer(params, "offset", 0)
    limit = _query_integer(params, "limit", 5_000)
    if offset < 0:
        raise HttpError(
            400,
            "invalid_trace_offset",
            "offset 必须是非负整数",
            message_en="offset must be a non-negative integer",
        )
    if not 1 <= limit <= 5_000:
        raise HttpError(
            400,
            "invalid_trace_limit",
            "limit 必须是 1 到 5000 之间的整数",
            message_en="limit must be an integer between 1 and 5000",
        )
    return batch_id, offset, limit


def _visualization_query_options(query: str) -> Dict[str, int]:
    """Parse bounded trace paging without changing the scenario JSON body."""

    params = parse_qs(query, keep_blank_values=True)
    allowed = {
        "trace_offset",
        "trace_limit",
        "memory_segment_limit",
    }
    unexpected = sorted(set(params) - allowed)
    if unexpected:
        raise HttpError(
            400,
            "invalid_trace_query",
            "仿真运行包含未知的可视化查询参数：{}".format(
                ", ".join(unexpected)
            ),
            message_en="unknown visualization query parameters: {}".format(
                ", ".join(unexpected)
            ),
        )
    offset = _query_integer(params, "trace_offset", 0)
    limit = _query_integer(
        params, "trace_limit", DEFAULT_VISUALIZATION_EVENT_LIMIT
    )
    memory_limit = _query_integer(
        params,
        "memory_segment_limit",
        DEFAULT_VISUALIZATION_MEMORY_SEGMENT_LIMIT,
    )
    if offset < 0:
        raise HttpError(
            400,
            "invalid_trace_offset",
            "trace_offset 必须是非负整数",
            message_en="trace_offset must be a non-negative integer",
        )
    if not 1 <= limit <= MAX_VISUALIZATION_EVENT_LIMIT:
        raise HttpError(
            400,
            "invalid_trace_limit",
            "trace_limit 必须在 1 到 {} 之间".format(
                MAX_VISUALIZATION_EVENT_LIMIT
            ),
            message_en="trace_limit must be between 1 and {}".format(
                MAX_VISUALIZATION_EVENT_LIMIT
            ),
        )
    if not 1 <= memory_limit <= MAX_VISUALIZATION_MEMORY_SEGMENT_LIMIT:
        raise HttpError(
            400,
            "invalid_memory_segment_limit",
            "memory_segment_limit 必须在 1 到 {} 之间".format(
                MAX_VISUALIZATION_MEMORY_SEGMENT_LIMIT
            ),
            message_en="memory_segment_limit must be between 1 and {}".format(
                MAX_VISUALIZATION_MEMORY_SEGMENT_LIMIT
            ),
        )
    return {
        "visualization_offset": offset,
        "visualization_limit": limit,
        "visualization_memory_segment_limit": memory_limit,
    }


def _url_host(host: str) -> str:
    return "[{}]".format(host) if ":" in host and not host.startswith("[") else host


__all__ = [
    "HeteroLLMRequestHandler",
    "HeteroLLMThreadingHTTPServer",
    "build_server",
    "ensure_valid_or_http_error",
    "scenario_to_payload",
    "serve",
    "static_asset",
    "validation_payload",
    "validation_payload_for_scenario",
]
