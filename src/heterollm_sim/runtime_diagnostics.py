"""Runtime environment inspection and local exception diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as distribution_version
import os
from pathlib import Path
import platform as platform_module
import struct
import sys
import threading
import traceback
from typing import Any, Dict, Optional, Tuple, Union
from uuid import uuid4


_DIAGNOSTIC_LOG_ENV = "HETEROLLM_SIM_DIAGNOSTIC_LOG"
_LOG_WRITE_LOCK = threading.Lock()


def health_payload(program_version: str) -> Dict[str, Any]:
    """Return the stable health contract plus detailed runtime capabilities."""

    ortools = ortools_diagnostics()
    available_solvers = ["auto", "builtin"]
    if ortools["available"]:
        available_solvers.append("ortools")
    return {
        "ok": True,
        "service": "heterollm-sim",
        "version": program_version,
        "runtime": {
            "python_version": platform_module.python_version(),
            "executable": sys.executable,
            "architecture_bits": struct.calcsize("P") * 8,
            "platform": platform_module.platform(),
        },
        "ortools": ortools,
        "available_solvers": available_solvers,
    }


def ortools_diagnostics() -> Dict[str, Any]:
    """Inspect the OR-Tools package and run a minimal deterministic CP-SAT solve."""

    try:
        package_version: Optional[str] = distribution_version("ortools")
    except PackageNotFoundError:
        package_version = None
        metadata_error = None
    except Exception as exc:  # pragma: no cover - unusual package metadata failure
        package_version = None
        metadata_error = _exception_message_zh("读取 OR-Tools 包版本失败", exc)
    else:
        metadata_error = None

    try:
        cp_model = import_module("ortools.sat.python.cp_model")
    except Exception as exc:
        return {
            "available": False,
            "version": package_version,
            "cp_sat_available": False,
            "error": _exception_message_zh("导入 OR-Tools CP-SAT 模块失败", exc),
            "error_type": type(exc).__name__,
            "probe_ok": False,
        }

    probe_ok, probe_error, probe_error_type = probe_cp_sat(cp_model)
    error = probe_error or metadata_error
    error_type = probe_error_type
    if error_type is None and metadata_error is not None:
        error_type = "PackageMetadataError"
    return {
        "available": bool(probe_ok),
        "version": package_version,
        "cp_sat_available": True,
        "error": error,
        "error_type": error_type,
        "probe_ok": bool(probe_ok),
    }


def probe_cp_sat(cp_model: Optional[Any] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    """Solve ``value == 1`` to verify that CP-SAT and its native runtime work."""

    try:
        module = cp_model or import_module("ortools.sat.python.cp_model")
        model = module.CpModel()
        value = model.NewBoolVar("runtime_probe")
        model.Add(value == 1)
        solver = module.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        status = solver.Solve(model)
        if status != module.OPTIMAL or solver.Value(value) != 1:
            try:
                status_name = solver.StatusName(status)
            except Exception:
                status_name = str(status)
            raise RuntimeError("最小模型未得到最优解，求解状态为 {}".format(status_name))
    except Exception as exc:
        return (
            False,
            _exception_message_zh("OR-Tools CP-SAT 最小求解失败", exc),
            type(exc).__name__,
        )
    return True, None, None


def diagnostic_log_path(
    explicit_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> Path:
    """Resolve the local diagnostics log, allowing an explicit or env override."""

    if explicit_path is not None:
        return Path(explicit_path).expanduser()
    configured = os.environ.get(_DIAGNOSTIC_LOG_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home().joinpath(".heterollm-sim", "diagnostics.log")


def record_unexpected_exception(
    exc: BaseException,
    *,
    context: str,
    log_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> str:
    """Print and append a full traceback, returning a response-safe diagnostic ID."""

    diagnostic_id = uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()
    traceback_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    header = (
        "[{}] 未预料异常，诊断 ID={}，上下文={}\n".format(
            timestamp, diagnostic_id, context
        )
    )

    # The terminal remains the reliable fallback even when the log directory is
    # unavailable or read-only.
    print(header, end="", file=sys.stderr)
    print(
        traceback_text,
        end="" if traceback_text.endswith("\n") else "\n",
        file=sys.stderr,
    )

    try:
        resolved_path = diagnostic_log_path(log_path)
        with _LOG_WRITE_LOCK:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with resolved_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(header)
                handle.write(traceback_text)
                if not traceback_text.endswith("\n"):
                    handle.write("\n")
                handle.write("\n")
        print("完整诊断已写入：{}".format(resolved_path), file=sys.stderr)
    except Exception as log_exc:  # pragma: no cover - environment-specific I/O
        print(
            "诊断日志写入失败（{}）：{}".format(
                type(log_exc).__name__, str(log_exc) or "未提供异常消息"
            ),
            file=sys.stderr,
        )
    return diagnostic_id


def format_doctor_report(payload: Dict[str, Any]) -> str:
    """Format a concise Chinese report for the CLI doctor command."""

    runtime = payload["runtime"]
    ortools = payload["ortools"]
    lines = [
        "HeteroLLM Simulator 运行环境诊断",
        "程序版本：{}".format(payload["version"]),
        "Python 版本：{}".format(runtime["python_version"]),
        "Python 可执行文件：{}".format(runtime["executable"]),
        "Python 位数：{} 位".format(runtime["architecture_bits"]),
        "平台：{}".format(runtime["platform"]),
        "OR-Tools 包版本：{}".format(ortools["version"] or "未检测到"),
        "CP-SAT 模块导入：{}".format(
            "成功" if ortools["cp_sat_available"] else "失败"
        ),
        "CP-SAT 最小求解：{}".format("成功" if ortools["probe_ok"] else "失败"),
        "可用求解器：{}".format("、".join(payload["available_solvers"])),
    ]
    if ortools["error"]:
        lines.append("诊断错误：{}".format(ortools["error"]))
    lines.append("诊断结果：{}".format("通过" if ortools["available"] else "未通过"))
    return "\n".join(lines)


def _exception_message_zh(action: str, exc: BaseException) -> str:
    detail = str(exc).strip() or "异常未提供详细消息"
    return "{}（{}）：{}".format(action, type(exc).__name__, detail)


__all__ = [
    "diagnostic_log_path",
    "format_doctor_report",
    "health_payload",
    "ortools_diagnostics",
    "probe_cp_sat",
    "record_unexpected_exception",
]
