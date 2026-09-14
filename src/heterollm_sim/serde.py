"""Canonical serialization helpers for reproducible run manifests."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict


def to_primitive(value: Any) -> Any:
    """Convert simulator dataclasses and enums into JSON-compatible values."""

    if value is None:
        return None

    value_type = type(value)
    if value_type is str or value_type is int or value_type is bool:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("JSON 数值必须是有限值，不能使用 NaN 或 Infinity")
        return value

    if is_dataclass(value):
        result = {}
        for field in fields(value):
            item = getattr(value, field.name)
            if item is None and field.metadata.get("omit_none"):
                continue
            result[field.name] = to_primitive(item)
        return result
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key.value) if isinstance(key, Enum) else str(key): to_primitive(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON 数值必须是有限值，不能使用 NaN 或 Infinity")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError("unsupported serialization value: {!r}".format(type(value)))


def canonical_json(value: Any, *, indent: int = 2) -> str:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        separators=(",", ": "),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    payload = canonical_json(value, indent=None).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    def reject_non_finite(token: str) -> None:
        raise ValueError(
            "JSON 数值必须是有限值，不能使用 {}".format(token)
        )

    data = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_non_finite,
    )
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    return data
