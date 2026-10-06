"""Architecture dispatch used by the GGUF model importer.

The native llama.cpp loader does not select an implementation from the model
brand.  It reads ``general.architecture`` from GGUF and dispatches that ID to
an architecture implementation.  Several IDs can share one implementation
when their tensor layout and forward graph are compatible.  This small
registry mirrors that boundary for the simulator: it keeps GGUF identification
separate from the code that materializes :class:`LayerSpec` and
:class:`ModelGraph`.

The registry describes *graph families*, rather than individual checkpoints.
``llama``, ``qwen2`` and dense ``qwen3`` therefore share the same standard
decoder adapter, while Qwen3.5 is routed to the hybrid decoder implementation
because its linear-attention blocks have a different graph contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class GGUFArchitectureAdapter:
    """Static dispatch metadata for one GGUF architecture implementation."""

    adapter_id: str
    architecture_ids: Tuple[str, ...]
    graph_family: str
    # ``hybrid`` marks graph construction that has a non-standard sequence
    # mixer (currently Qwen3.5's gated-delta-net blocks).
    hybrid: bool = False

    def matches(self, architecture: object) -> bool:
        return str(architecture or "").strip().lower() in self.architecture_ids


_ADAPTERS: Tuple[GGUFArchitectureAdapter, ...] = (
    GGUFArchitectureAdapter(
        adapter_id="llama_like_decoder",
        # Qwen3 dense checkpoints use the same decoder tensor contract as
        # Llama/Qwen2.  Qwen3 MoE is intentionally not included: its expert
        # tensor layout needs a separate adapter before it can be executable.
        architecture_ids=("llama", "qwen2", "qwen3"),
        graph_family="decoder_transformer",
    ),
    GGUFArchitectureAdapter(
        adapter_id="qwen35_hybrid_decoder",
        architecture_ids=("qwen35", "qwen35moe"),
        graph_family="qwen3_5_hybrid_transformer",
        hybrid=True,
    ),
)

_BY_ARCHITECTURE: Dict[str, GGUFArchitectureAdapter] = {
    architecture: adapter
    for adapter in _ADAPTERS
    for architecture in adapter.architecture_ids
}


def resolve_gguf_architecture_adapter(
    architecture: object,
) -> GGUFArchitectureAdapter | None:
    """Return the adapter selected by GGUF ``general.architecture``.

    The lookup is deliberately exact after lower-casing.  Silently guessing
    a graph for an unknown architecture would produce plausible-looking but
    incorrect simulation costs, so callers should fail closed when this
    function returns ``None``.
    """

    return _BY_ARCHITECTURE.get(str(architecture or "").strip().lower())


def list_gguf_architecture_adapters() -> Tuple[GGUFArchitectureAdapter, ...]:
    """Return the immutable registry in deterministic order."""

    return _ADAPTERS


__all__ = [
    "GGUFArchitectureAdapter",
    "list_gguf_architecture_adapters",
    "resolve_gguf_architecture_adapter",
]
