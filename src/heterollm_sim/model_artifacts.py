"""Persistent, graph-native model artifacts used by the simulator.

The simulator's executable model is the V4 ``ModelSpec.graph`` object.  This
module stores that object in a small JSON envelope so a frontend can save a
model graph and later run a scenario by artifact id.  It deliberately does
not pretend to be a GGUF writer: a graph contains logical weights and shape
contracts, while a GGUF file also contains binary weight payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Optional
from uuid import uuid4

from .serde import canonical_json, read_json, to_primitive


MODEL_ARTIFACT_SCHEMA = "heterollm.model-artifact/v1"
_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class ModelArtifactError(ValueError):
    """Raised when a saved model artifact is missing or invalid."""


def default_model_artifact_dir() -> Path:
    """Return the per-user model-artifact directory without creating it."""

    configured = os.environ.get("HETEROLLM_SIM_MODEL_ARTIFACT_DIR")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        if root:
            return Path(root) / "heterollm-sim" / "model-artifacts"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "heterollm-sim" / "model-artifacts"
    return Path.home() / ".cache" / "heterollm-sim" / "model-artifacts"


def validate_artifact_id(value: Any) -> str:
    if not isinstance(value, str) or not _ARTIFACT_ID.fullmatch(value):
        raise ModelArtifactError("artifact_id 必须是安全的模型文件标识符")
    return value


def _model_payload(value: Any) -> dict[str, Any]:
    """Normalize a ModelSpec or model JSON object and validate its graph."""

    from .config import model_from_dict

    raw = to_primitive(value)
    if not isinstance(raw, Mapping):
        raise ModelArtifactError("模型文件的 model 必须是对象")
    model = model_from_dict(dict(raw))
    if not model.graph.executable:
        raise ModelArtifactError("只能保存可执行模型图；当前图仅包含未验证的证据结构")
    return to_primitive(model)


def _artifact_path(artifact_id: str, artifact_dir: Optional[Path]) -> Path:
    safe_id = validate_artifact_id(artifact_id)
    root = Path(artifact_dir) if artifact_dir is not None else default_model_artifact_dir()
    return root / (safe_id + ".model.json")


def _new_artifact_id(model: Mapping[str, Any]) -> str:
    name = str(model.get("name") or "model").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")[:48] or "model"
    return "{}-{}".format(slug, uuid4().hex[:12])


def make_model_artifact(
    model: Any,
    *,
    artifact_id: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build and validate an artifact document without writing it."""

    normalized_model = _model_payload(model)
    selected_id = validate_artifact_id(artifact_id) if artifact_id else _new_artifact_id(normalized_model)
    return {
        "schema_version": MODEL_ARTIFACT_SCHEMA,
        "artifact_id": selected_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": normalized_model,
        "provenance": to_primitive(dict(provenance or {})),
    }


def save_model_artifact(
    model: Any,
    *,
    artifact_dir: Optional[Path] = None,
    artifact_id: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Validate and atomically save one executable model artifact."""

    document = make_model_artifact(
        model,
        artifact_id=artifact_id,
        provenance=provenance,
    )
    target = _artifact_path(document["artifact_id"], artifact_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(document) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(target)
    except Exception:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return dict(document, path=str(target.resolve()))


def load_model_artifact(
    artifact_id: str,
    *,
    artifact_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Read, validate, and return one saved artifact document."""

    target = _artifact_path(artifact_id, artifact_dir)
    try:
        document = read_json(target)
    except FileNotFoundError as exc:
        raise ModelArtifactError("未找到模型文件：{}".format(artifact_id)) from exc
    except (OSError, ValueError) as exc:
        raise ModelArtifactError("模型文件无法读取：{}".format(artifact_id)) from exc
    if document.get("schema_version") != MODEL_ARTIFACT_SCHEMA:
        raise ModelArtifactError("不支持的模型文件 schema_version")
    if document.get("artifact_id") != artifact_id:
        raise ModelArtifactError("模型文件 artifact_id 与文件名不一致")
    model = _model_payload(document.get("model"))
    result = dict(document)
    result["model"] = model
    result["path"] = str(target.resolve())
    return result


def list_model_artifacts(*, artifact_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """List valid-looking local artifact summaries without loading graphs."""

    root = Path(artifact_dir) if artifact_dir is not None else default_model_artifact_dir()
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.model.json")):
        try:
            document = read_json(path)
            if document.get("schema_version") != MODEL_ARTIFACT_SCHEMA:
                continue
            artifact_id = validate_artifact_id(document.get("artifact_id"))
            model = document.get("model")
            if not isinstance(model, Mapping):
                continue
            graph = model.get("graph")
            graph_attributes = graph.get("attributes", {}) if isinstance(graph, Mapping) else {}
            architecture = graph_attributes.get("architecture", "") if isinstance(graph_attributes, Mapping) else ""
            items.append({
                "artifact_id": artifact_id,
                "name": str(model.get("name") or artifact_id),
                "architecture": str(architecture or ""),
                "schema_version": MODEL_ARTIFACT_SCHEMA,
                "path": str(path.resolve()),
                "provenance": to_primitive(document.get("provenance", {})),
            })
        except (OSError, ValueError, TypeError):
            continue
    return items


__all__ = [
    "MODEL_ARTIFACT_SCHEMA",
    "ModelArtifactError",
    "default_model_artifact_dir",
    "validate_artifact_id",
    "make_model_artifact",
    "save_model_artifact",
    "load_model_artifact",
    "list_model_artifacts",
]
