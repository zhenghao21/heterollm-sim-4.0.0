"""Dependency-free dtype and quantization bit-width parsing."""

from __future__ import annotations

import re
from typing import Optional, Tuple


_DTYPE_BITS = {
    "f32": 32,
    "fp32": 32,
    "float32": 32,
    "f16": 16,
    "fp16": 16,
    "float16": 16,
    "bf16": 16,
    "bfloat16": 16,
    "fp8": 8,
    "float8": 8,
    "int8": 8,
    "uint8": 8,
    "int4": 4,
    "uint4": 4,
}

_QUANTIZATION_SEPARATORS = re.compile(r"[-_\s]+")

_CANONICAL_DTYPES = {
    "f32": "fp32",
    "fp32": "fp32",
    "float32": "fp32",
    "f16": "fp16",
    "fp16": "fp16",
    "float16": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp8": "fp8",
    "float8": "fp8",
    "int8": "int8",
    "uint8": "uint8",
    "int4": "int4",
    "uint4": "uint4",
}


def canonical_dtype(dtype_name: str) -> str:
    """Normalize aliases while preserving representation distinctions."""

    normalized = dtype_name.strip().lower().replace("-", "").replace("_", "")
    return _CANONICAL_DTYPES.get(normalized, normalized)


def dtype_bits(dtype_name: str, *, unsupported_message: str) -> int:
    normalized = dtype_name.lower().replace("-", "").replace("_", "")
    bits = _DTYPE_BITS.get(normalized)
    if bits is None:
        raise ValueError(unsupported_message)
    return bits


def layer_precision_bits(
    dtype_name: str,
    quantization: Optional[str],
    *,
    unsupported_dtype_message: str,
    unsupported_quantization_message: Optional[str] = None,
) -> Tuple[int, int]:
    parsed = _quantization_bits(
        quantization,
        unsupported_message=(
            unsupported_quantization_message
            or "unsupported quantization {}".format(quantization)
        ),
    )
    if parsed is not None and parsed[0] is not None:
        return parsed[0], parsed[1]
    bits = dtype_bits(
        dtype_name,
        unsupported_message=unsupported_dtype_message,
    )
    return bits, parsed[1] if parsed is not None else bits


def weight_storage_bits(
    dtype_name: str,
    quantization: Optional[str],
    *,
    unsupported_dtype_message: str,
    unsupported_quantization_message: str,
) -> int:
    parsed = _quantization_bits(
        quantization,
        unsupported_message=unsupported_quantization_message,
    )
    if parsed is not None:
        return parsed[1]
    return dtype_bits(
        dtype_name,
        unsupported_message=unsupported_dtype_message,
    )


def _quantization_bits(
    quantization: Optional[str], *, unsupported_message: str
) -> Optional[Tuple[Optional[int], int]]:
    """Parse the declared W/A bit-width contract without guessing a scheme."""

    if quantization is None:
        return None
    normalized = _QUANTIZATION_SEPARATORS.sub("", quantization.strip().lower())
    if not normalized:
        return None
    match = re.fullmatch(r"w(\d+)(?:a(\d+))?", normalized)
    if match is None:
        raise ValueError(unsupported_message)
    weight_bits = int(match.group(1))
    activation_bits = (
        int(match.group(2)) if match.group(2) is not None else None
    )
    if weight_bits <= 0 or (
        activation_bits is not None and activation_bits <= 0
    ):
        raise ValueError(unsupported_message)
    return activation_bits, weight_bits
