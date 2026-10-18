"""Read an active scan and publish compact live data without running simulations.

Snapshot: python scripts/b200_scan_dashboard.py --output-dir DIR --snapshot
Publisher: pass {"token": "..."} as one stdin line; never put tokens in arguments.
The publisher writes only its own status file, not scan results or progress.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import getpass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request


SEMANTICS = "canonical_shared_resident_weights_v1"
SUCCESS = {"completed", "forecast_completed"}
TERMINAL = SUCCESS | {"forecast_rejected", "capacity_rejected", "failed"}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/b200_hbm_hbf_0_5b_7b_fast_20261018"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def numeric(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def process_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class JsonCache:
    """Reuse large JSON documents until mtime/size change; tolerate atomic writes."""

    def __init__(self) -> None:
        self.entries: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
        self.parse_count = 0

    def read(self, path: Path, warnings: list[str]) -> dict[str, Any] | None:
        previous = self.entries.get(path)
        for _ in range(2):
            try:
                stat = path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                if previous and previous[0] == signature:
                    return previous[1]
                value = json.loads(path.read_text(encoding="utf-8-sig"))
                self.parse_count += 1
                after = path.stat()
                if signature != (after.st_mtime_ns, after.st_size):
                    continue
                if not isinstance(value, dict):
                    raise ValueError("not an object")
                self.entries[path] = signature, value
                return value
            except FileNotFoundError:
                self.entries.pop(path, None)
                return None  # Archived or absent rows must not remain active.
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        warnings.append(f"{path.parent.name}/{path.name} 正在发布或暂时无法读取" + ("；沿用上次成功读取" if previous else ""))
        return previous[1] if previous else None


BUDGET_LABELS = {
    "forecast_tail_ns": "预测尾段",
    "prefix_erase_bias_guard_ns": "前缀擦除偏差",
    "future_erase_serial_guard_ns": "未来普通擦除",
    "prefix_gc_bias_guard_ns": "前缀 GC 复制偏差",
    "prefix_write_placement_bias_guard_ns": "前缀写入放置偏差",
    "prefix_read_placement_bias_guard_ns": "前缀读取放置偏差",
    "future_gc_erase_serial_guard_ns": "未来 GC 擦除",
    "future_gc_non_erase_serial_guard_ns": "未来 GC 复制",
    "future_write_placement_serial_guard_ns": "未来写入放置",
    "future_read_placement_serial_guard_ns": "未来读取放置",
    "future_gc_carry_in_serial_guard_ns": "GC 进入尾段的剩余耗时",
    "prefix_gc_read_remap_symmetric_bias_guard_ns": "前缀 KV 重映射不确定性",
    "future_gc_read_remap_symmetric_guard_ns": "未来 KV 重映射不确定性",
    "prefix_gc_persistent_cache_symmetric_bias_guard_ns": "前缀持久缓存不确定性",
    "future_gc_persistent_cache_symmetric_guard_ns": "未来持久缓存不确定性",
    "prefix_gc_host_state_symmetric_bias_guard_ns": "前缀 HOST 状态不确定性",
    "future_gc_host_state_symmetric_guard_ns": "未来 HOST 状态不确定性",
    "prefix_maintenance_total_guard_ns": "前缀维护合计",
    "future_maintenance_total_guard_ns": "未来维护合计",
    "symmetric_maintenance_guard_ns": "双向维护不确定性合计",
    "conditional_lower_maintenance_guard_ns": "条件区间下界维护项",
    "conditional_upper_maintenance_guard_ns": "条件区间上界维护项",
    "legacy_future_erase_serial_guard_ns": "旧公式擦除参考值（不叠加）",
}


def budget_items(approximation: dict[str, Any]) -> list[dict[str, Any]]:
    breakdown = approximation.get("error_budget_breakdown") or {}
    return [{"key": key, "label": BUDGET_LABELS.get(key, key), "ns": value}
            for key, value in breakdown.items()
            if key.endswith("_ns") and numeric(value) is not None]


def compact_point(spec: dict[str, Any], row: dict[str, Any] | None, model: dict[str, Any]) -> dict[str, Any]:
    row = row or {}
    state = row.get("status", "pending")
    approx = row.get("approximation") or {}
    decision = approx.get("paired_budget_decision") or {}
    reason = row.get("reason")
    if not reason and state in {"forecast_rejected", "failed", "capacity_rejected"}:
        reason = approx.get("budget_unavailable_reason") or "; ".join(str(v) for v in decision.get("reasons", [])) or None
    result = {
        **{key: spec[key] for key in ("point_id", "model_id", "architecture", "prompt_tokens", "output_tokens")},
        "model_name": model.get("label", spec["model_id"]), "parameters_b": numeric(model.get("parameters_b")),
        "status": state, "ttft_ms": None, "tpot_ms": None,
        "wall_seconds": numeric(row.get("wall_seconds")),
        "real_prefill_steps": numeric(row.get("real_prefill_steps")),
        "real_decode_steps": numeric(row.get("real_decode_steps")),
        "sampled_decode_steps": numeric(approx.get("sampled_decode_steps")),
        "predicted_decode_steps": numeric(approx.get("predicted_decode_steps")),
        "single_error_bound": numeric(decision.get("candidate_relative_error_budget", approx.get("decode_conditional_relative_error_budget"))),
        "reference_error_bound": numeric(decision.get("reference_relative_error_budget")),
        "pair_error_bound": numeric(decision.get("pair_relative_error_budget")),
        "paired_accepted": decision.get("accepted") if isinstance(decision.get("accepted"), bool) else None,
        "reason": reason, "started_at": row.get("started_at"), "finished_at": row.get("finished_at"),
        "budget": budget_items(approx),
        "assumptions": [str(value) for value in approx.get("model_assumptions", [])],
    }
    if state in SUCCESS:
        for source, target in (("ttft_ns", "ttft_ms"), ("tpot_ns", "tpot_ms")):
            value = numeric(row.get(source))
            result[target] = value / 1e6 if value is not None and value > 0 else None
    return result


def _older(value: Any, cutoff: Any) -> bool:
    try:
        return bool(value and cutoff and datetime.fromisoformat(value) < datetime.fromisoformat(cutoff))
    except (ValueError, TypeError):
        return False


class ScanReader:
    def __init__(self, output_dir: Path, alive: Callable[[Any], bool] = process_alive) -> None:
        self.output_dir = output_dir.resolve()
        self.cache = JsonCache()
        self.alive = alive

    def snapshot(self) -> dict[str, Any]:
        warnings: list[str] = []
        manifest = self.cache.read(self.output_dir / "manifest.json", warnings)
        if not manifest or manifest.get("physical_weight_semantics") != SEMANTICS:
            raise ValueError("manifest missing or incompatible physical weight semantics")
        status = self.cache.read(self.output_dir / "status.json", warnings) or {}
        if status and status.get("physical_weight_semantics") != SEMANTICS:
            warnings.append("状态文件的物理权重语义不匹配；只显示当前语义的结果")
            status = {}
        models = {model["model_id"]: model for model in manifest["models"]}
        specs = manifest["points"]
        known = {spec["point_id"]: spec for spec in specs}
        rows = []
        for spec in specs:
            identifier = spec["point_id"]
            if Path(identifier).name != identifier or "/" in identifier or "\\" in identifier:
                raise ValueError("unsafe point identifier")
            row = self.cache.read(self.output_dir / "points" / f"{identifier}.json", warnings)
            if row and (row.get("physical_weight_semantics") != SEMANTICS or any(row.get(k) != spec[k] for k in (
                    "point_id", "model_id", "architecture", "prompt_tokens", "output_tokens"))):
                warnings.append(f"{identifier} 的语义或配置不匹配，未计入当前结果")
                row = None
            rows.append(compact_point(spec, row, models.get(spec["model_id"], {})))
        by_id = {row["point_id"]: row for row in rows}
        parameters = manifest.get("parameters") or {}
        chunk = parameters.get("prefill_chunk_tokens", 2048)
        if not isinstance(chunk, int) or chunk <= 0:
            raise ValueError("invalid prefill chunk size")
        decode_limit = manifest.get("simulation", {}).get("fast_decode", {}).get("max_sample_tokens", 128)
        running = []
        for item in status.get("running", []):
            identifier = item.get("point_id")
            if identifier not in known or by_id[identifier]["status"] in TERMINAL:
                continue
            progress = self.cache.read(self.output_dir / "progress" / f"{identifier}.json", warnings) or {}
            if (progress.get("physical_weight_semantics") != SEMANTICS or _older(progress.get("updated_at"), item.get("started_at"))):
                progress = {}
            event = progress.get("progress") or {}
            actual_pid = progress.get("pid")  # Manager running.pid is a Windows launcher, never the actual worker.
            if actual_pid is None:
                warnings.append(f"{identifier} 尚无真实 worker PID；不把启动器当作 worker")
            alive = self.alive(actual_pid) if actual_pid is not None else False
            if actual_pid is not None and not alive:
                warnings.append(f"{identifier} 的实际 worker 未存活；结果仍以扫描管理器为准")
            detail = {
                **{key: known[identifier][key] for key in ("point_id", "model_id", "architecture", "prompt_tokens", "output_tokens")},
                "pid": actual_pid, "worker_alive": alive,
                "prefill_completed": numeric(progress.get("real_prefill_steps")) or 0,
                "prefill_total": math.ceil(known[identifier]["prompt_tokens"] / chunk),
                "decode_completed": numeric(progress.get("real_decode_steps")) or 0,
                "decode_limit": decode_limit, "stage": event.get("stage", progress.get("status", "starting")),
                "detail_completed": numeric(event.get("completed")), "detail_total": numeric(event.get("total")),
                "wall_seconds": numeric(progress.get("wall_seconds")) or 0,
                "updated_at": progress.get("updated_at"), "simulated_time_ns": numeric(event.get("simulated_time_ns")),
            }
            running.append(detail)
            # A partial request never gains whole-request TTFT/TPOT, even if a stale result once existed.
            active = compact_point(known[identifier], {**progress, "status": "running", "started_at": item.get("started_at")}, models.get(item.get("model_id"), {}))
            by_id[identifier].update(active)
        counts = {state: sum(row["status"] == state for row in rows) for state in TERMINAL}
        completed = sum(counts[state] for state in SUCCESS)
        hbf = parameters.get("hbf_parameters") or {}
        geometry = hbf.get("geometry") or {}
        settings = [
            {"label": "GPU / 请求", "value": "B200 · 单 GPU · 单请求 · BF16"},
            {"label": "仿真", "value": f"Prefill 物理执行 · adaptive Decode ≤{decode_limit} 真实步 · 条件误差目标 10%"},
            {"label": "精度说明", "value": "条件预算不等于完整详细模式实测误差保证；TPOT 包含预测尾段"},
            {"label": "HBM", "value": "HBM3E；权重、KV、激活与工作区放在 HBM"},
            {"label": "HBF", "value": f"零 HBM；OCP 0.7.0 协议 · FlashLabAE {hbf.get('array_units', 256)} bank 研究几何"},
            {"label": "HBF 几何", "value": f"{geometry.get('channels', 16)} 通道 × {geometry.get('banks_per_die', 16)} bank · {hbf.get('page_bytes', 4096)} B/page"},
            {"label": "研究假设", "value": "HBF 时延和每通道在途限制为研究假设，非 OCP 规定或产品实测值"},
            {"label": "CPU / HOST DRAM", "value": "保留，参与物理执行与 staging"},
            {"label": "物理权重语义", "value": SEMANTICS},
        ]
        return {
            "schema": "b200-scan-dashboard/v1", "scan_id": self.output_dir.name,
            "observed_at": utc_now(), "status": {
                "state": status.get("status", "unknown"), "total": len(specs), "completed": completed,
                "forecast_completed": counts["forecast_completed"], "forecast_rejected": counts["forecast_rejected"],
                "capacity_rejected": counts["capacity_rejected"], "failed": counts["failed"],
                "pending": len(specs) - sum(counts.values()) - len(running),
                "manager_alive": self.alive(status.get("manager_pid")),
                "stop_requested": (self.output_dir / "stop.request").exists(), "updated_at": status.get("updated_at"),
                "reason": status.get("reason"), "retry_group_completed": status.get("retry_group_completed"),
                "retry_group_total": status.get("retry_group_total"), "retry_group_accepted": status.get("retry_group_accepted"),
                "running": running,
            },
            "models": [{"id": value["model_id"], "label": value.get("label", value["model_id"]), "parameters_b": value.get("parameters_b")} for value in manifest["models"]],
            "settings": settings, "points": rows, "warnings": warnings,
        }


class SinglePublisher:
    """OS-released lock prevents a duplicate publisher after a crash or restart."""

    def __enter__(self):
        self.handle = (Path(tempfile.gettempdir()) / "heterollm_b200_dashboard_publisher.lock").open("a+b")
        self.handle.seek(0)
        self.handle.write(b" ")
        self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("another scan dashboard publisher is already running") from None
        return self

    def __exit__(self, *_):
        self.handle.close()


def endpoint(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}):
        raise ValueError("publisher requires HTTPS (HTTP allowed only for local QA)")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("publisher URL must not contain credentials, query parameters, or fragments")
    path = parsed.path.rstrip("/")
    if not path.endswith("/api/scan"):
        path += "/api/scan"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def transport_proxies(url: str) -> dict[str, str]:
    if urllib.parse.urlsplit(url).hostname in {"127.0.0.1", "localhost"}:
        return {}
    proxies = {key: value for key, value in urllib.request.getproxies().items()
               if key in {"http", "https", "all", "no"} and value}
    # NODE_USE_ENV_PROXY is interpreted as a proxy by urllib. Together with
    # NO_PROXY it can prevent Windows' existing system proxy from being read.
    if os.name == "nt" and not any(proxies.get(key) for key in ("http", "https", "all")):
        registry = urllib.request.getproxies_registry()
        proxies = {**registry, **proxies}
    return proxies


def send_snapshot(url: str, token: str, snapshot: dict[str, Any]) -> int:
    payload = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="PUT", headers={
        "Content-Type": "application/json", "OAI-Sites-Authorization": "Bearer " + token,
        "User-Agent": "B200-Memory-Lab/1.0", "Accept": "application/json",
    })
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, newurl):
            raise urllib.error.HTTPError(request.full_url, code, "redirect refused", headers, fp)

    with urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler(transport_proxies(url))).open(request, timeout=20) as response:
        if response.headers.get_content_type() != "application/json":
            raise ValueError("publisher expected a JSON acknowledgement")
        acknowledgement = json.loads(response.read(65536))
        if not isinstance(acknowledgement, dict) or acknowledgement.get("ok") is not True:
            raise ValueError("publisher expected an affirmative acknowledgement")
        return response.status


def write_publisher_status(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(values, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def start_background(args: argparse.Namespace, token: str, destination: str) -> dict[str, Any]:
    """Detach only the uploader; pass credentials through its private stdin pipe."""
    log_path = ROOT / "tmp/b200_scan_dashboard_publisher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()),
               "--output-dir", str(args.output_dir.resolve()), "--url", destination,
               "--interval", str(args.interval), "--status-file", str(args.status_file.resolve())]
    if args.once:
        command.append("--once")
    options: dict[str, Any] = {"cwd": str(ROOT), "stdin": subprocess.PIPE,
                               "stderr": subprocess.STDOUT, "close_fds": True}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        options["start_new_session"] = True
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, **options)
        try:
            process.stdin.write((json.dumps({"token": token}) + "\n").encode("utf-8"))
            process.stdin.flush()
        finally:
            process.stdin.close()
    return {"state": "background_started", "launcher_pid": process.pid, "log": str(log_path),
            "status_file": str(args.status_file.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url")
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--background", action="store_true", help="detach uploader after reading the credential from stdin")
    parser.add_argument("--snapshot", action="store_true", help="emit compact JSON only; no token, upload or status file")
    parser.add_argument("--status-file", type=Path, default=ROOT / "tmp/b200_scan_dashboard_publisher.json")
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("--interval must be at least 1 second")
    if args.snapshot and args.background:
        parser.error("--snapshot cannot be combined with --background")
    reader = ScanReader(args.output_dir)
    if args.snapshot:
        print(json.dumps(reader.snapshot(), ensure_ascii=True, allow_nan=False, separators=(",", ":")))
        return 0
    if not args.url:
        parser.error("--url is required unless --snapshot is used")
    destination = endpoint(args.url)
    try:
        line = getpass.getpass("Service credential JSON (hidden): ") if sys.stdin.isatty() else sys.stdin.readline()
        credentials = json.loads(line)
        token = credentials.get("token")
        if not isinstance(token, str) or not token or "\r" in token or "\n" in token:
            raise ValueError()
    except (ValueError, AttributeError):
        parser.error('stdin must contain one JSON line with a nonempty "token"')
    del credentials, line
    if args.status_file.resolve().is_relative_to(args.output_dir.resolve()):
        parser.error("--status-file must be outside the read-only scan directory")
    if args.background:
        print(json.dumps(start_background(args, token, destination), ensure_ascii=True), flush=True)
        return 0
    failures = 0
    last_success = None
    with SinglePublisher():
        while True:
            started = time.monotonic()
            state = {"pid": os.getpid(), "updated_at": utc_now(), "last_success_at": last_success,
                     "consecutive_failures": failures, "http_status": None, "error": None,
                     "scan_id": args.output_dir.name, "state": "publishing"}
            try:
                snapshot = reader.snapshot()
                code = send_snapshot(destination, token, snapshot)
                if not 200 <= code < 300:
                    raise ValueError("unexpected HTTP response")
                failures = 0
                last_success = utc_now()
                state.update(state="healthy", http_status=code, last_success_at=last_success,
                             consecutive_failures=0, completed=snapshot["status"]["completed"])
            except urllib.error.HTTPError as error:
                failures += 1
                # Never log response bodies, request headers or arbitrary exception text containing credentials.
                state.update(state="error", http_status=error.code,
                             error="HTTP 401: service authorization rejected" if error.code == 401 else f"HTTP {error.code}: upload rejected")
            except (urllib.error.URLError, TimeoutError, OSError):
                failures += 1
                state.update(state="error", error="network unavailable or request timed out")
            except (ValueError, KeyError, TypeError):
                failures += 1
                state.update(state="error", error="scan data or publisher configuration is invalid")
            state.update(updated_at=utc_now(), consecutive_failures=failures)
            write_publisher_status(args.status_file, state)
            print(json.dumps(state, ensure_ascii=True), flush=True)
            if args.once:
                return 0 if not failures else 2
            delay = max(args.interval, min(120, 5 * (2 ** min(failures, 5)))) if failures else args.interval
            time.sleep(max(0, delay - (time.monotonic() - started)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
