"""Opt-in fixed-source final-layer row selection, without scheduling or costs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


SOURCE_KEY = "llama_cpp_final_layer_output_selection"
_BACKEND_COMMIT = "0f3a71be15af836d277c9f918adfafb45732677e"
_ARCHITECTURES = {
    "qwen2_decoder": ("qwen2", "before_last_ffn"),
    "llama_decoder": ("llama", "before_last_ffn"),
    "qwen3_5_hybrid_transformer": ("qwen35", "after_final_norm"),
}


def source_declaration() -> dict[str, object]:
    """Return a fresh declaration for the proved ordinary completion branch."""
    return {
        "schema_version": "heterollm.llama-final-layer-output-selection/v1",
        "backend_commit": _BACKEND_COMMIT,
        "request_kind": "completion",
        "explicit_logits": True,
        "embeddings": False,
        "embeddings_nextn_masked": False,
        "speculative_type": "none",
    }


def _architecture(graph_architecture: str) -> Tuple[str, str]:
    if not isinstance(graph_architecture, str) or graph_architecture not in _ARCHITECTURES:
        raise ValueError("final-layer selection requires a supported exact graph architecture")
    return _ARCHITECTURES[graph_architecture]


@dataclass(frozen=True)
class FinalLayerOutputSelection:
    """One invocation's selected lane indices; empty indices mean proved zero rows."""

    graph_architecture: str
    token_rows: int
    selected_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        _architecture(self.graph_architecture)
        if type(self.token_rows) is not int or self.token_rows <= 0:
            raise ValueError("token_rows must be a positive integer")
        if not isinstance(self.selected_indices, tuple):
            raise ValueError("selected_indices must be an immutable tuple")
        previous = -1
        for index in self.selected_indices:
            if type(index) is not int or not previous < index < self.token_rows:
                raise ValueError("selected indices must be integers in range, strictly increasing and unique")
            previous = index

    @property
    def position(self) -> str:
        return _architecture(self.graph_architecture)[1]

    @property
    def logit_rows(self) -> int:
        return len(self.selected_indices)

    @property
    def ffn_rows(self) -> int:
        return self.logit_rows if self.position == "before_last_ffn" else self.token_rows

    @property
    def final_norm_rows(self) -> int:
        return self.ffn_rows

    @property
    def source_gather_nodes(self) -> int:
        return 2 if self.position == "before_last_ffn" else 1

    @property
    def gather_count(self) -> int:
        """Nonempty GET_ROWS executions; zero-sized source nodes do not execute."""
        return self.source_gather_nodes if self.logit_rows else 0

    def audit_metadata(self) -> dict[str, object]:
        """Return fresh JSON-compatible facts without changing invocation identity."""
        return {
            "model": "fixed_source_final_layer_output_selection/v1",
            "backend_commit": _BACKEND_COMMIT,
            "graph_architecture": self.graph_architecture,
            "backend_architecture": _architecture(self.graph_architecture)[0],
            "position": self.position,
            "token_rows": self.token_rows,
            "selected_indices": list(self.selected_indices),
            "logit_rows": self.logit_rows,
            "ffn_rows": self.ffn_rows,
            "final_norm_rows": self.final_norm_rows,
            "gather_count": self.gather_count,
            "source_gather_nodes": self.source_gather_nodes,
            "scope": "ordinary completion with explicit output flags; MTP and embeddings excluded",
            "source_work_only": True,
        }


@dataclass(frozen=True)
class FinalLayerOutputPolicy:
    """Architecture-derived policy; callers cannot override the selection position."""

    graph_architecture: str

    def __post_init__(self) -> None:
        _architecture(self.graph_architecture)

    @property
    def position(self) -> str:
        return _architecture(self.graph_architecture)[1]

    def select(self, *, token_rows: int, selected_indices: Tuple[int, ...]) -> FinalLayerOutputSelection:
        return FinalLayerOutputSelection(self.graph_architecture, token_rows, selected_indices)


def resolve_declaration(
    raw: object, graph_architecture: str, *, mtp_present: bool,
) -> Optional[FinalLayerOutputPolicy]:
    """Validate explicit metadata; absence returns before introducing any new checks.

    The planner supplies ``mtp_present = scenario.workload.mtp is not None``:
    even a present but disabled MTP object is outside this source declaration.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(SOURCE_KEY + " must be a declaration mapping or None")
    expected = source_declaration()
    if set(raw) != set(expected):
        raise ValueError(SOURCE_KEY + " must contain exactly the fixed declaration fields")
    for field, value in expected.items():
        if type(raw[field]) is not type(value) or raw[field] != value:
            raise ValueError("unsupported final-layer source declaration field: " + field)
    if type(mtp_present) is not bool:
        raise ValueError("mtp_present must be a boolean derived from MTP object presence")
    if mtp_present:
        raise ValueError("final-layer source selection excludes every present MTP configuration")
    return FinalLayerOutputPolicy(graph_architecture)


__all__ = (
    "SOURCE_KEY", "source_declaration", "resolve_declaration",
    "FinalLayerOutputPolicy", "FinalLayerOutputSelection",
)
