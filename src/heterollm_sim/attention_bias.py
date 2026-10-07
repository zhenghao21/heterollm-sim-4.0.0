"""Explicit ordinary-attention projection bias, independent of model names."""

from collections.abc import Mapping
from dataclasses import dataclass

from .cost_models import ElementwiseWorkload


SCHEMA = "heterollm.attention-qkv-bias/v1"


@dataclass(frozen=True)
class AttentionQKVBias:
    query_elements: int
    key_elements: int
    value_elements: int
    storage_bits: int
    source: str

    @property
    def weight_bytes(self) -> int:
        return (self.query_elements + self.key_elements + self.value_elements) * self.storage_bits // 8


def resolve_attention_qkv_bias(metadata: Mapping, *, query_width: int, kv_width: int):
    if "attention_qkv_bias" not in metadata:
        return None
    raw = metadata["attention_qkv_bias"]
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SCHEMA:
        raise ValueError("attention_qkv_bias requires an explicit supported schema")
    for key, expected in (("query_elements", query_width), ("key_elements", kv_width), ("value_elements", kv_width)):
        if type(raw.get(key)) is not int or raw[key] <= 0 or raw[key] != expected:
            raise ValueError("attention_qkv_bias {} differs from projection width".format(key))
    if type(raw.get("storage_bits")) is not int or raw["storage_bits"] not in (16, 32):
        raise ValueError("attention_qkv_bias requires 16-bit or 32-bit floating-point storage")
    if not isinstance(raw.get("source"), str) or not raw["source"].strip():
        raise ValueError("attention_qkv_bias requires source provenance")
    result = AttentionQKVBias(query_width, kv_width, kv_width, raw["storage_bits"], raw["source"])
    if "weight_bytes" in raw and (type(raw["weight_bytes"]) is not int or raw["weight_bytes"] != result.weight_bytes):
        raise ValueError("attention_qkv_bias weight_bytes differs from the declared vectors")
    return result


def projection_bias_workload(*, tokens: int, local_width: int, activation_bits: int, bias_bits: int, name: str):
    """A broadcast vector is read once per invocation, not once per token."""
    for key, value in (("tokens", tokens), ("local_width", local_width)):
        if type(value) is not int or value <= 0:
            raise ValueError("{} must be a positive integer".format(key))
    if activation_bits not in (16, 32) or bias_bits not in (16, 32):
        raise ValueError("projection bias requires explicit floating-point storage widths")
    activation_bytes = tokens * local_width * activation_bits // 8
    bias_bytes = local_width * bias_bits // 8
    return ElementwiseWorkload(
        elements=tokens * local_width, operations_per_element=1,
        input_count=2, input_bits=activation_bits, output_bits=activation_bits,
        read_storage_bytes=activation_bytes + bias_bytes,
        write_storage_bytes=activation_bytes,
        name=name,
    )
